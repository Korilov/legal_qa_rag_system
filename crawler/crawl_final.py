import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import re
import time
import os
import urllib3
from docx import Document

urllib3.disable_warnings()

# ================= 动态路径配置 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSONL = os.path.join(BASE_DIR, "data", "raw", "law_articles.jsonl")
TEMP_DIR = os.path.join(BASE_DIR, "data", "raw", "temp_docx")
os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://flk.npc.gov.cn/",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
}


# ================= 创建带重试机制的 Session =================
def create_session():
    session = requests.Session()
    # 如果请求失败（如超时、502），自动重试 5 次，每次间隔 2 秒
    retry = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(HEADERS)
    return session


session = create_session()


# ================= 核心函数 =================

def fetch_list(keyword="", page=1, size=10):
    url = "https://flk.npc.gov.cn/law-search/search/list"
    payload = {
        "searchRange": 1, "searchType": 2, "searchContent": keyword,
        "pageNum": page, "pageSize": size,
        "orderByParam": {"order": "-1", "sort": ""},
        "flfgCodeId": [], "zdjgCodeId": [], "sxx": [],
        "gbrq": [], "sxrq": [], "gbrqYear": [], "xgzlSearch": False,
    }
    try:
        # 超时时间放宽到 60 秒
        resp = session.post(url, json=payload, verify=False, timeout=60)
        resp.raise_for_status()
        return resp.json().get("rows", [])
    except Exception as e:
        print(f"获取列表失败: {e}")
        return []


def download_docx(bbbs, save_path):
    api = f"https://flk.npc.gov.cn/law-search/download/pc?format=docx&bbbs={bbbs}"
    try:
        # 获取签名 URL
        resp = session.get(api, verify=False, timeout=60)
        resp.raise_for_status()
        url = resp.json()["data"]["url"]

        # 下载文件，时间放宽到 120 秒
        file_resp = session.get(url, verify=False, timeout=120)
        file_resp.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(file_resp.content)
        return True
    except Exception as e:
        print(f"  [下载失败] {e}")
        return False


def parse_docx(path):
    try:
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    except Exception:
        return ""


def split_articles(full_text):
    if not full_text: return []
    pattern = re.compile(r"(第[一二三四五六七八九十百千零〇0-9]+条(?:之[一二三四五六七八九十])?)")
    matches = list(pattern.finditer(full_text))
    articles = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        article_no = m.group(1)
        body = full_text[start:end].strip()
        body = body[len(article_no):].strip(" ：:　\n")
        if len(body) >= 5:
            articles.append({"article_no": article_no, "text": body})
    if not articles and len(full_text) > 20:
        articles.append({"article_no": "全文", "text": full_text[:3000]})
    return articles


# ================= 主流程 =================

def crawl(keyword="", max_pages=1, page_size=5):  # 先测试抓5个
    # 改为追加模式 'a'，如果之前有数据就不会覆盖
    fout = open(OUT_JSONL, "a", encoding="utf-8")
    total_articles = 0

    for page in range(1, max_pages + 1):
        print(f"\n正在获取第 {page} 页列表...")
        rows = fetch_list(keyword=keyword, page=page, size=page_size)

        if not rows:
            print("获取不到数据，可能是网络问题或接口限制。")
            break

        for item in rows:
            bbbs = item.get("bbbs")
            title = re.sub(r"<[^>]+>", "", item.get("title", "")).strip()
            if not bbbs or not title: continue

            print(f"处理中: {title}")
            docx_path = os.path.join(TEMP_DIR, f"{bbbs}.docx")

            if not os.path.exists(docx_path):
                if not download_docx(bbbs, docx_path):
                    time.sleep(2)  # 失败也等一下
                    continue
                time.sleep(1.5)  # 成功下载后休息一下，防封

            full_text = parse_docx(docx_path)
            if not full_text: continue

            articles = split_articles(full_text)

            for a in articles:
                rec = {
                    "law_name": title,
                    "article_no": a["article_no"],
                    "text": a["text"],
                    "category": item.get("flxz", ""),
                    "publish_date": item.get("gbrq", ""),
                    "effective_date": item.get("sxrq", "") or "",
                    "bbbs": bbbs
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_articles += 1
            print(f"  -> 成功提取 {len(articles)} 条")

    fout.close()
    print(f"\n全部完成！本次共写入 {total_articles} 条法条。")


if __name__ == "__main__":
    # 这里我们先测试抓取“劳动合同”的第1页的5部法规
    crawl(keyword="劳动合同", max_pages=20, page_size=20)