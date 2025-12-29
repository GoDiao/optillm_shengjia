import re
import time
import json
import math, random
import os, pathlib, shutil, tempfile, time
from typing import Tuple, List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import quote_plus

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SLUG = "web_search"

CERT_FILE="./HW_IT_CAs/huawei-ca-bundle.pem"

def _ensure_dir(p: str, mode: int = 0o700) -> str:
    d = pathlib.Path(p); d.mkdir(parents=True, exist_ok=True)
    try: d.chmod(mode)
    except: pass
    return str(d)

import re

# 认为“可能是二进制/乱码”的启发式：非打印控制字符占比 > 10%
_CONTROL_RE = re.compile(r"[\x00-\x08\x0B-\x1F\x7F]")

def looks_binary(text: str) -> bool:
    if not text:
        return False
    bad = sum((ord(c) < 32 and c not in "\t\r\n") or ord(c) == 127 for c in text)
    return bad / max(1, len(text)) > 0.10

def clean_text(s: str, limit: int) -> str:
    s = _CONTROL_RE.sub("", s or "")
    if len(s) > limit:
        s = s[:limit] + "..."
    return s

def extract_markdown(sess, headers, url: str, timeout_s: int = 25) -> str:
    """命中 PDF/疑似乱码时，调用 Tavily Extract，把内容转成 markdown 可读文本。"""
    try:
        payload = {
            "urls": [url],
            "format": "markdown",
            "extract_depth": "advanced",
        }
        r = sess.post(
            "https://api.tavily.com/extract",
            data=json.dumps(payload),
            headers=headers,
            timeout=timeout_s,
            verify=CERT_FILE
        )
        
        # CHECK FOR CREDIT EXHAUSTION
        check_tavily_credits(r)
        
        if r.status_code >= 400:
            print(f"[TAVILY] extract {r.status_code}: {r.text[:200]}")
            return ""
        data = r.json()
        items = data.get("results") or []
        if not items:
            return ""
        text = (items[0].get("raw_content") or "").strip()
        return text
    except TavilyCreditsExhausted:
        raise  # Re-raise to propagate up
    except Exception as e:
        print(f"[TAVILY] extract error: {e}")
        return ""


def extract_text(sess, headers, url: str, timeout_s: int = 25) -> str:
    try:
        payload = {"urls": [url], "format": "text", "extract_depth": "advanced"}
        r = sess.post(
            "https://api.tavily.com/extract",
            data=json.dumps(payload),
            headers=headers,
            timeout=timeout_s,
            verify=CERT_FILE
        )
        if r.status_code >= 400:
            print(f"[TAVILY] extract(text) {r.status_code}: {r.text[:200]}")
            return ""
        data = r.json()
        items = data.get("results") or []
        return (items[0].get("raw_content") or "").strip() if items else ""
    except Exception as e:
        print(f"[TAVILY] extract(text) error: {e}")
        return ""


class TavilyError(Exception):
    pass

class TavilyCreditsExhausted(Exception):
    """Raised when Tavily API credits are exhausted"""
    pass

def check_tavily_credits(response):
    """Check if Tavily API response indicates exhausted credits"""
    status = response.status_code
    
    # Common status codes for credit exhaustion
    if status == 402:  # Payment Required
        raise TavilyCreditsExhausted(
            "Tavily API credits exhausted (402 Payment Required). "
            "Please add credits at https://tavily.com"
        )
    
    if status == 429:  # Too Many Requests / Rate Limited
        try:
            data = response.json()
            error_msg = data.get("error", "").lower()
            if "credit" in error_msg or "quota" in error_msg or "limit" in error_msg:
                raise TavilyCreditsExhausted(
                    f"Tavily API credits exhausted (429): {data.get('error', 'Rate limit exceeded')}. "
                    "Please add credits at https://tavily.com"
                )
        except (ValueError, KeyError):
            pass
    
    # Check error message in response body
    try:
        data = response.json()
        error_msg = data.get("error", "").lower()
        if any(keyword in error_msg for keyword in ["credit", "quota exceeded", "insufficient credits", "balance"]):
            raise TavilyCreditsExhausted(
                f"Tavily API credits exhausted: {data.get('error', 'Unknown error')}. "
                "Please add credits at https://tavily.com"
            )
    except (ValueError, KeyError):
        pass

def _build_session(timeout_s: int = 20) -> requests.Session:
    """requests session，带重试；自动继承 http_proxy/https_proxy/no_proxy 环境变量。"""
    s = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.request_timeout = timeout_s
    return s


def tavily_web_search(
    query: str,
    num_results: int = 10,
    max_len: int = 6000,
    delay_seconds: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    使用 Tavily Web Search API 搜索，并返回与你原先 Google 爬虫相同的结构：
    [{"title": str, "url": str, "snippet": str}, ...]
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        print("[TAVILY] Missing TAVILY_API_KEY env")
        return []

    # Tavily 的单次最大返回条数通常在 10 左右，这里做保护
    max_per_call = max(1, min(10, num_results or 10))
    want = max(1, num_results or 10)

    sess = _build_session(timeout_s=25)
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 你可以根据需要调整这些参数（保持结果简洁，避免超长 context）
    base_payload = {
        "query": query,
        "max_results": max_per_call,
        "search_depth": "advanced",          # "basic" / "advanced"
        "include_answer": False,          # 不要让它拼答案，避免长文本
        "include_raw_content": False,     # 仅返回精简字段
        "include_images": False,          # 避免返回图片列表导致 context 爆长
        "include_image_descriptions": False,
        # 需要的话可加 include_domains / exclude_domains
        # "include_domains": ["site1.com", "site2.com"],
        # "exclude_domains": ["x.com"]
        "min_score_threshold": 0.5,
        "max_content_length_per_page": max_len,
    }

    results: List[Dict[str, str]] = []
    seen = set()

    # 计算需要多少次调用（如果用户要的 >10）
    rounds = math.ceil(want / max_per_call)

    for r in range(rounds):
        try:
            payload = dict(base_payload)
            # 若不是第一次，可以轻微扰动查询，帮助去重（可选）
            if r > 0:
                payload["query"] = f"{query} {''.join([' ' for _ in range(r)])}"

            resp = sess.post(
                "https://api.tavily.com/search",
                data=json.dumps(payload),
                headers=headers,
                timeout=sess.request_timeout,
                verify=CERT_FILE
            )
            if resp.status_code == 401:
                print("[TAVILY] Unauthorized: check TAVILY_API_KEY")
                return []
            if resp.status_code >= 500:
                print(f"[TAVILY] Server error {resp.status_code}: {resp.text[:200]}")
                continue

            data = resp.json()
            raw_items = data.get("results") or []

            for it in raw_items:
                url = (it.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                title = (it.get("title") or "").strip()
                if payload["include_raw_content"]:
                    snippet = (it.get("raw_content") or it.get("snippet") or "").strip()
                else:
                    snippet = (it.get("content") or it.get("snippet") or "").strip()

                # —— 响应侧增强：PDF/疑似乱码 → Extract 转 markdown —— #
                need_extract = url.lower().endswith(".pdf") or looks_binary(snippet)

                if need_extract:
                    md = extract_markdown(sess, headers, url, timeout_s=sess.request_timeout)
                    if md:
                        snippet = md  # 用转换后的可读文本替换
                    # 若抽取失败，仍然使用原片段（但会在 clean_text 里清洗/截断）

                # 统一清洗 + 截断，避免控制字符/过长污染上下文
                snippet = clean_text(snippet, max_len)

                ## 简短化 snippet，避免爆 context
                #if len(snippet) > max_len:
                #    print("search result content %d exceed max len of %d\n", len(snippet), max_len)
                #    snippet = snippet[:max_len] + "..."
                results.append({"title": title or url, "url": url, "snippet": snippet})
                if len(results) >= want:
                    break

            if len(results) >= want:
                break

            # 轻微限速（如果调用多轮）
            time.sleep(0.4 + random.random() * 0.6)

        except requests.RequestException as e:
            print(f"[TAVILY] RequestException: {e}")
            continue
        except Exception as e:
            print(f"[TAVILY] Unexpected error: {e}")
            continue

    # 额外去重（保险）
    uniq, seen2 = [], set()
    for r in results:
        if r["url"] not in seen2:
            uniq.append(r)
            seen2.add(r["url"])

    # 可选限速：模仿你之前的 delay_seconds
    if delay_seconds and delay_seconds > 0:
        time.sleep(delay_seconds)

    return uniq[:want]


class BrowserSessionManager:
    """
    Manages a single browser session across multiple searches.
    Implements context manager for automatic cleanup.
    """
    def __init__(self, headless: bool = False, timeout: int = 30):
        self.headless = headless
        self.timeout = timeout
        self._searcher = None
        self._search_count = 0
        self._session_start_time = None
    
    def __enter__(self):
        """Context manager entry - ensures browser is ready"""
        self.get_or_create_searcher()
        self._session_start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures browser cleanup"""
        self.close()
        return False  # Don't suppress exceptions
    
    def get_or_create_searcher(self) -> 'GoogleSearcher':
        """Get existing searcher or create a new one"""
        if self._searcher is None or self._searcher.driver is None:
            if self._searcher is None:
                print("🌐 Creating new browser session for research...")
            else:
                print("🔄 Recreating browser session (previous session invalidated)...")
            self._searcher = GoogleSearcher(
                headless=self.headless,
                timeout=self.timeout
            )
        return self._searcher
    
    def search(self, query: str, num_results: int = 10, delay_seconds: Optional[int] = None) -> List[Dict[str, str]]:
        """Perform a search using the managed browser session with automatic recovery"""
        try:
            searcher = self.get_or_create_searcher()
            self._search_count += 1
            session_duration = time.time() - self._session_start_time if self._session_start_time else 0
            print(f"🔍 Search #{self._search_count} in current session (instance: {id(self)}, duration: {session_duration:.1f}s): {query[:50]}...")
            return searcher.search(query, num_results, delay_seconds)
        except Exception as e:
            error_msg = str(e).lower()
            # Check for session-related errors
            if 'invalid session id' in error_msg or 'session deleted' in error_msg:
                print("⚠️  Browser session invalidated, attempting recovery...")
                # Invalidate current searcher
                if self._searcher:
                    try:
                        self._searcher.close()
                    except:
                        pass  # Ignore errors during cleanup
                self._searcher = None
                
                # Try once more with a fresh session
                try:
                    searcher = self.get_or_create_searcher()
                    print("✅ New browser session created, retrying search...")
                    return searcher.search(query, num_results, delay_seconds)
                except Exception as retry_error:
                    print(f"❌ Session recovery failed: {str(retry_error)}")
                    return []  # Return empty results instead of crashing
            else:
                # For other errors, just log and return empty results
                print(f"❌ Search error: {str(e)}")
                return []
    
    def close(self):
        """Close the browser session"""
        if self._searcher is not None:
            try:
                self._searcher.close()
                if self._session_start_time:
                    duration = time.time() - self._session_start_time
                    print(f"🏁 Browser session closed after {self._search_count} searches ({duration:.1f}s)")
            except Exception as e:
                print(f"⚠️ Error closing browser session: {e}")
            finally:
                self._searcher = None
                self._search_count = 0
                self._session_start_time = None
    
    def is_active(self) -> bool:
        """Check if browser session is active"""
        return self._searcher is not None and self._searcher.driver is not None


class GoogleSearcher:
    def __init__(self, headless: bool = False, timeout: int = 30):
        self.timeout = timeout
        self.headless = headless
        self.driver = None
        self.setup_driver(headless)


    def setup_driver(self, headless: bool = False):
        """Setup Chrome driver with robust headless flags & local driver"""
        try:
            # 1) 解析二进制/driver（强制 stable，绕过 /opt 链）
            chrome_bin = os.environ.get("CHROME_BIN", "/usr/bin/google-chrome-stable")
            if not shutil.which(chrome_bin):
                # 兜底再试一次 /usr/bin/google-chrome
                if shutil.which("/usr/bin/google-chrome"):
                    chrome_bin = "/usr/bin/google-chrome"
                else:
                    raise RuntimeError(f"Chrome binary not found: {chrome_bin}")

            chromedriver = os.environ.get("CHROMEDRIVER", "/usr/local/bin/chromedriver")
            if not shutil.which(chromedriver):
                # 兜底：你本机 122 的缓存位置，如不在可删掉这分支
                maybe = os.path.expanduser("~/.wdm/drivers/chromedriver/linux64/122.0.6261.57/chromedriver-linux64/chromedriver")
                if shutil.which(maybe):
                    chromedriver = maybe
                else:
                    raise RuntimeError(f"Chromedriver not found: {chromedriver}")

            # 2) 每次一个独立 profile，避免锁/残留
            uid = os.getuid() if hasattr(os, "getuid") else 0
            xdg  = _ensure_dir(os.environ.get("XDG_RUNTIME_DIR", "/tmp/xdg"), 0o700)
            os.environ["XDG_RUNTIME_DIR"] = xdg
            os.environ.setdefault("TMPDIR", "/tmp")

            home = os.path.expanduser("~")
            prof = _ensure_dir(os.path.join(home, ".cache/chrome-profile"), 0o700)
            dcache = _ensure_dir(os.path.join(home, ".cache/chrome-cache"), 0o700)

            # 3) 组装 Chrome 启动参数（使用 pipe，避免 DevToolsActivePort 文件）
            opts = Options()
            opts.binary_location = os.environ.get("CHROME_BIN", "/opt/google/chrome/google-chrome")

            # 关键 flags（你已有）：--headless=new/--no-sandbox/--disable-dev-shm-usage/--remote-debugging-pipe/--user-data-dir 等
            args = [
                "--headless=new",
                "--disable-dev-shm-usage",
                "--remote-debugging-pipe",
                f"--user-data-dir={prof}",
                f"--disk-cache-dir={dcache}",
                "--disable-extensions",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-networking",
                "--metrics-recording-only",
                "--mute-audio",
                "--hide-scrollbars",
                "--window-size=1280,800",
                "--disable-software-rasterizer",
                "--disable-features=TranslateUI,AudioServiceOutOfProcess,OptimizationHints",
                "--ignore-certificate-errors",
            ]

            # 企业代理（从环境变量自动读取）
            proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
            if proxy:
                from urllib.parse import urlparse
                p = urlparse(proxy)
                if p.hostname and p.port:
                    proxy_addr = f"http://{p.hostname}:{p.port}"
                    opts.add_argument(f"--proxy-server={proxy_addr}")
                else:
                    print(f"Invalid proxy: {proxy}")
                #opts.add_argument(f"--proxy-server={proxy}")
                no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
                bypass = ["localhost", "127.0.0.1", "::1", "<local>"]
                bypass += [t.strip() for t in no_proxy.replace(";", ",").split(",") if t.strip() and "/" not in t]
                opts.add_argument(f"--proxy-bypass-list={','.join(dict.fromkeys(bypass))}")
                   
            # 合并从环境传入的调试参数（--enable-logging=stderr --v=1）
            extra = os.environ.get("CHROME_ARGS", "").split()
            args.extend([a for a in extra if a])
            
            for a in args:
                opts.add_argument(a)
    
            # 4) 启动，并打印关键信息便于核对
            service = Service(executable_path=chromedriver, log_path="/tmp/chromedriver.log")
            self.driver = webdriver.Chrome(service=service, options=opts)
            self.driver.set_page_load_timeout(self.timeout)

            print("[Chrome OK]")
            print("  binary:", chrome_bin)
            print("  driver:", chromedriver)
            print("  profile:", prof)
            print("  args:", args)

        except Exception as e:
            raise Exception(f"Failed to setup Chrome driver: {e}")

    
    #def setup_driver(self, headless: bool = False):
    #    """Setup Chrome driver with appropriate options"""
    #    try:
    #        chrome_options = Options()
    #        if headless:
    #            chrome_options.add_argument("--headless")
    #        else:
    #            # Non-headless mode - position window for visibility
    #            chrome_options.add_argument("--window-size=1280,800")
    #            chrome_options.add_argument("--window-position=100,100")
    #        
    #        # Common options
    #        chrome_options.add_argument("--no-sandbox")
    #        chrome_options.add_argument("--disable-dev-shm-usage")
    #        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    #        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    #        chrome_options.add_experimental_option('useAutomationExtension', False)
    #        
    #        # More human-like settings
    #        chrome_options.add_argument("--disable-gpu")
    #        chrome_options.add_argument("--disable-web-security")
    #        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
    #        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    #        
    #        # Use webdriver-manager to automatically manage ChromeDriver
    #        service = Service(ChromeDriverManager().install())
    #        self.driver = webdriver.Chrome(service=service, options=chrome_options)
    #        self.driver.set_page_load_timeout(self.timeout)
    #    except Exception as e:
    #        raise Exception(f"Failed to setup Chrome driver: {str(e)}")

    def _dbg_dump(self, tag: str):
        try:
            ts = int(time.time())
            base = f"/tmp/{tag}-{ts}"
            # 1) URL & Title
            print(f"[DBG] URL: {self.driver.current_url}")
            print(f"[DBG] Title: {self.driver.title}")
            # 2) SELECTOR COUNTS
            sel_counts = {
                "#search a h3": len(self.driver.find_elements(By.CSS_SELECTOR, "#search a h3")),
                ".yuRUbf > a": len(self.driver.find_elements(By.CSS_SELECTOR, ".yuRUbf > a")),
                "div.g": len(self.driver.find_elements(By.CSS_SELECTOR, "div.g")),
            }
            print(f"[DBG] Selector counts: {sel_counts}")
            # 3) 截图 + HTML
            png = f"{base}.png"
            html = f"{base}.html"
            self.driver.save_screenshot(png)
            with open(html, "w", encoding="utf-8") as f:
                f.write(self.driver.page_source)
            print(f"[DBG] Saved {png} & {html}")
            # 4) Console 日志（需开启 performance logging 或 --enable-logging）
            try:
                for entry in self.driver.get_log("browser"):
                    print(f"[CONSOLE] {entry}")
            except Exception:
                pass
        except Exception as e:
            print(f"[DBG] dump failed: {e}")
    
    def _handle_google_consent(self, timeout=6):
        # 任何 get() 后都调用
        try:
            # 先看页面内按钮
            for sel in [
                (By.ID, "L2AGLb"),
                (By.CSS_SELECTOR, "button[aria-label*='Accept' i]"),
                (By.XPATH, "//button[contains(., 'I agree') or contains(., 'Accept all')]"),
            ]:
                try:
                    btn = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable(sel))
                    btn.click()
                    WebDriverWait(self.driver, 2).until(EC.invisibility_of_element(btn))
                    return True
                except: pass
            # 再看 iframe
            ifr = self.driver.find_elements(By.CSS_SELECTOR, "iframe[src*='consent']")
            for f in ifr:
                try:
                    self.driver.switch_to.frame(f)
                    for sel in [
                        (By.ID, "L2AGLb"),
                        (By.CSS_SELECTOR, "button[aria-label*='Accept' i]"),
                        (By.XPATH, "//button[contains(., 'I agree') or contains(., 'Accept all')]"),
                    ]:
                        try:
                            btn = WebDriverWait(self.driver, 2).until(EC.element_to_be_clickable(sel))
                            btn.click()
                            self.driver.switch_to.default_content()
                            return True
                        except: pass
                    self.driver.switch_to.default_content()
                except:
                    try: self.driver.switch_to.default_content()
                    except: pass
        except TimeoutException:
            pass
        return False
    
    def _looks_like_captcha(self):
        html = self.driver.page_source.lower()
        url = (self.driver.current_url or "").lower()
        return ("recaptcha" in html or "/sorry/" in url or "unusual traffic" in html)
        
    def detect_captcha(self) -> bool:
        """Detect if CAPTCHA is present on the page"""
        try:
            # Check for common CAPTCHA indicators
            page_source = self.driver.page_source.lower()
            captcha_indicators = [
                'recaptcha',
                'captcha',
                'are you a robot',
                'not a robot',
                'unusual traffic',
                'automated requests',
                'verify you\'re human',
                'verify that you\'re not a robot'
            ]
            
            for indicator in captcha_indicators:
                if indicator in page_source:
                    print("INFO: CAPTCHA indicator: ", indicator)
                    print(page_source)
                    return True
            
            # Check for reCAPTCHA iframe
            try:
                self.driver.find_element(By.CSS_SELECTOR, "iframe[src*='recaptcha']")
                print("INFO: CAPTCHA iframe: ", By.CSS_SELECTOR)
                return True
            except:
                pass
            
            # Check for CAPTCHA challenge div
            try:
                self.driver.find_element(By.ID, "captcha")
                print("INFO: CAPTCHA ID: ", By.ID)
                return True
            except:
                pass
            
            return False
        except:
            return False
    
    def wait_for_captcha_resolution(self, max_wait: int = 300) -> bool:
        """Wait for CAPTCHA to be resolved with user confirmation"""
        print("🚨 CAPTCHA DETECTED! 🚨")
        print("Please solve the CAPTCHA in the browser window.")
        print("After solving the CAPTCHA, press ENTER here to continue...")
        
        if self.headless:
            print("ERROR: CAPTCHA detected in headless mode - cannot solve automatically")
            return False
        
        # Wait for user to press Enter after solving CAPTCHA
        try:
            input("Press ENTER after you have solved the CAPTCHA: ")
        except KeyboardInterrupt:
            print("\\nSearch cancelled by user")
            return False
        
        # Give a moment for the page to update after CAPTCHA resolution
        print("Checking if CAPTCHA has been resolved...")
        time.sleep(2)
        
        # Verify CAPTCHA is actually resolved
        for attempt in range(3):
            if not self.detect_captcha():
                print("✅ CAPTCHA resolved successfully!")
                return True
            else:
                print(f"CAPTCHA still detected (attempt {attempt + 1}/3)")
                if attempt < 2:
                    response = input("CAPTCHA still present. Try again? Press ENTER to continue or 'q' to quit: ")
                    if response.lower() == 'q':
                        return False
                    time.sleep(2)
        
        print("❌ CAPTCHA still not resolved after 3 attempts")
        return False
    
    def search(self, query: str, num_results: int = 10, delay_seconds: Optional[int] = None) -> List[Dict[str, str]]:
        """Perform Google search and return results"""
        if not self.driver:
            raise Exception("Chrome driver not initialized")
        
        try:
            print(f"Searching for: {query}")
            if not self.headless:
                print("Browser window opened")
            
            # First navigate to Google homepage
            self.driver.get("https://www.google.com")
            self._dbg_dump("after-google-home")
            self._handle_google_consent()
            
            # Wait for page to load and check for CAPTCHA
            time.sleep(1)
            
            # Check if we hit a CAPTCHA immediately
            if self.detect_captcha():
                self._dbg_dump("first-captcha-detected")
                self._handle_google_consent()
                if not self.wait_for_captcha_resolution():
                    return []
            
            # Check for consent form or cookie banner and accept if present
            try:
                accept_button = self.driver.find_element(By.XPATH, "//button[contains(text(), 'Accept') or contains(text(), 'I agree') or contains(text(), 'Agree')]")
                accept_button.click()
                time.sleep(1)
            except:
                pass  # No consent form
            
            # Find search box and enter query
            try:
                # Try multiple selectors for the search box
                search_box = None
                for selector in [(By.NAME, "q"), (By.CSS_SELECTOR, "input[type='text']"), (By.CSS_SELECTOR, "textarea[name='q']")]:
                    try:
                        search_box = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located(selector)
                        )
                        break
                    except:
                        continue
                
                if search_box:
                    # Use ActionChains for more reliable input
                    actions = ActionChains(self.driver)
                    actions.move_to_element(search_box)
                    actions.click()
                    actions.pause(0.5)
                    # Clear existing text
                    search_box.clear()
                    actions.send_keys(query)
                    actions.pause(0.5)
                    actions.send_keys(Keys.RETURN)
                    actions.perform()
                    
                    # Wait briefly for page to start loading
                    time.sleep(1)
                    
                    # Check for CAPTCHA after search submission
                    if self.detect_captcha():
                        self._dbg_dump("second-captcha-detected")
                        self._handle_google_consent()
                        if not self.wait_for_captcha_resolution():
                            return []
                else:
                    raise Exception("Could not find search box")
            except:
                # Fallback to direct URL navigation
                print("Using direct URL navigation...")
                search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}&hl=en&gl=us&pws=0&gws_rd=cr"
                #search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num_results}"
                self.driver.get(search_url)
                self._dbg_dump("after-search-url")
                self._handle_google_consent()
 
                time.sleep(1)
                
                # Check for CAPTCHA on direct navigation
                if self.detect_captcha():
                    if not self.wait_for_captcha_resolution():
                        return []

            # Wait for search results
            wait = WebDriverWait(self.driver, 10)
            try:
                wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g, [data-sokoban-container], div[data-async-context]"))
                )
            except TimeoutException:
                # Check if it's a CAPTCHA page
                self._dbg_dump("timeout-waiting-serp")
                if self.detect_captcha():
                    if self.wait_for_captcha_resolution():
                        # Try waiting for results again after CAPTCHA
                        try:
                            wait.until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
                            )
                        except:
                            print("No results found after CAPTCHA resolution")
                            return []
                    else:
                        return []
                else:
                    print("Timeout waiting for search results")
                    return []


            results = []
            
            # Apply delay AFTER search results are loaded (to prevent triggering anti-bot measures)
            if delay_seconds is None:
                delay_seconds = random.randint(4, 32)  # Updated range: 4-32 seconds
            
            if delay_seconds > 0:
                print(f"Applying {delay_seconds} second delay after search...")
                time.sleep(delay_seconds)
            
            print("Extracting search results...")
            
            # Wait for search results to be present
            try:
                print("Waiting for search results to load...")
                # Wait for either the search container or the results themselves
                WebDriverWait(self.driver, 10).until(
                    lambda driver: driver.find_elements(By.CSS_SELECTOR, "div.g") or 
                                   driver.find_element(By.ID, "search") or
                                   driver.find_elements(By.CSS_SELECTOR, "[data-sokoban-container]")
                )
            except TimeoutException:
                print("Timeout waiting for search results. Checking for CAPTCHA...")
                if self.detect_captcha():
                    if not self.wait_for_captcha_resolution():
                        return []
                    # Try waiting again after CAPTCHA
                    try:
                        WebDriverWait(self.driver, 10).until(
                            lambda driver: driver.find_elements(By.CSS_SELECTOR, "div.g")
                        )
                    except:
                        print("Still no results after CAPTCHA resolution")
                        return []
                else:
                    print("No CAPTCHA detected, but timeout occurred - search may have failed")
                    return []
            
            # Debug: Print current URL and page title
            print(f"Current URL: {self.driver.current_url}")
            print(f"Page title: {self.driver.title}")
            
            # Extract search results - try multiple selectors
            search_results = []
            
            # First try the standard div.g selector
            search_results = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            print(f"Found {len(search_results)} results with div.g")
            
            # If no results, try alternative selectors
            if not search_results:
                # Try finding any element with data-hveid attribute (Google result containers)
                all_elements = self.driver.find_elements(By.CSS_SELECTOR, "[data-hveid]")
                print(f"Found {len(all_elements)} elements with data-hveid")
                
                # Filter to only those that have both h3 and a tags
                for elem in all_elements:
                    try:
                        # Check if this element has both h3 and a link
                        h3 = elem.find_element(By.TAG_NAME, "h3")
                        link = elem.find_element(By.CSS_SELECTOR, "a[href]")
                        if h3 and link:
                            search_results.append(elem)
                    except:
                        continue
                
                print(f"Filtered to {len(search_results)} valid result elements")
            
            if not search_results:
                print("No search results found with any method")
                # Debug: print some page source to see what we're getting
                print("Page source sample (first 500 chars):")
                print(self.driver.page_source[:500])
                return []
            
            # Limit processing to requested number of results
            results_to_process = min(len(search_results), num_results)
            print(f"Processing {results_to_process} results...")
            
            for i, result in enumerate(search_results[:results_to_process]):
                try:
                    # Skip if we already have enough results
                    if len(results) >= num_results:
                        break
                    
                    # Use the same extraction logic as getstars
                    try:
                        url = result.find_element(By.CSS_SELECTOR, "a").get_attribute("href")
                        title = result.find_element(By.CSS_SELECTOR, "h3").text
                        
                        # Skip Google internal URLs
                        if not url or "google.com" in url:
                            continue
                            
                        # Try to get snippet
                        snippet = ""
                        try:
                            # Try multiple snippet selectors
                            snippet_selectors = [".VwiC3b", ".aCOpRe", ".IsZvec"]
                            for selector in snippet_selectors:
                                try:
                                    snippet_elem = result.find_element(By.CSS_SELECTOR, selector)
                                    if snippet_elem and snippet_elem.text:
                                        snippet = snippet_elem.text
                                        break
                                except:
                                    pass
                        except:
                            pass
                        
                        # Add result
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet or "No description available"
                        })
                        
                        print(f"Extracted result {len(results)}: {title[:50]}...")
                        
                    except NoSuchElementException:
                        print(f"Failed to parse result {i+1}")
                        continue
                        
                except Exception as e:
                    # Skip problematic results
                    continue
            
            # Deduplicate results by URL
            seen_urls = set()
            unique_results = []
            for result in results:
                if result["url"] not in seen_urls:
                    seen_urls.add(result["url"])
                    unique_results.append(result)
            
            print(f"Successfully extracted {len(unique_results)} unique search results (from {len(results)} total)")
            
            return unique_results
            
        except TimeoutException as e:
            # Return empty results instead of raising
            print(f"Search timeout for query '{query}': {str(e)}")
            return []
        except WebDriverException as e:
            error_msg = str(e).lower()
            if 'invalid session id' in error_msg or 'session deleted' in error_msg:
                print(f"WebDriver session invalid: {str(e)}")
                # Invalidate the driver so next search creates a new one
                self.driver = None
            else:
                print(f"WebDriver error during search: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error during search: {str(e)}")
            return []
    
    def close(self):
        """Close the browser driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def extract_search_queries(text: str) -> List[str]:
    """Extract potential search queries from the input text"""
    # Clean up common prefixes from chat messages
    text = text.strip()
    # Remove common role prefixes
    for prefix in ["User:", "user:", "User ", "user ", "Assistant:", "assistant:", "System:", "system:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    # Look for explicit search requests
    # Note: Removed period (.) from exclusion to allow queries like "Python 3.12" to work
    # Updated to require at least one non-whitespace character after the prefix
    search_patterns = [
        r"search for[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"find information about[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
        r"look up[:\s]+(\S[^\n]*?)(?:\s*\n|$)", 
        r"research[:\s]+(\S[^\n]*?)(?:\s*\n|$)",
    ]
    
    queries = []
    for pattern in search_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Clean up the match
            cleaned = match.strip()
            # Remove trailing quotes (single or double)
            cleaned = cleaned.rstrip('"\'')
            # Remove leading quotes if they exist
            cleaned = cleaned.lstrip('"\'')
            # Only add non-empty queries
            if cleaned:
                queries.append(cleaned)
    
    # If no explicit patterns, use the text as a search query
    if not queries:
        # Check if this is a search command with empty query (e.g., "search for" with nothing after)
        search_prefixes = ["search for", "find information about", "look up", "research"]
        text_lower = text.lower().strip()
        
        # Don't use fallback if it's just a search prefix with nothing meaningful after
        is_empty_search = any(
            text_lower.startswith(prefix) and 
            len(text_lower.replace(prefix, "").strip().strip('"\'')) < 2
            for prefix in search_prefixes
        )
        
        if not is_empty_search:
            # Remove question marks and clean up
            cleaned_query = text.replace("?", "").strip()
            # Remove quotes from the query
            cleaned_query = cleaned_query.strip('"\'')
            
            # If it looks like a question or search query, use it
            if cleaned_query and len(cleaned_query.split()) > 2:
                queries.append(cleaned_query)
            else:
                # Clean up the text to make it search-friendly
                cleaned_query = re.sub(r'[^\w\s\.]', ' ', text)  # Keep periods for version numbers
                cleaned_query = ' '.join(cleaned_query.split())
                # Remove quotes after regex cleaning
                cleaned_query = cleaned_query.strip('"\'')
                
                if len(cleaned_query) > 100:
                    # Take first 100 characters
                    cleaned_query = cleaned_query[:100].rsplit(' ', 1)[0]
                if cleaned_query and len(cleaned_query) > 2:  # Ensure minimum length
                    queries.append(cleaned_query)
    
    return queries

def format_search_results(query: str, results: List[Dict[str, str]]) -> str:
    """Format search results into readable text"""
    if not results:
        return f"No search results found for: {query}"
    
    formatted = f"Search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"{i}. **{result['title']}**\n"
        formatted += f"   URL: {result['url']}\n"
        if result['snippet']:
            formatted += f"   Summary: {result['snippet']}\n"
        formatted += "\n"
    
    return formatted

def format_search_results_lmc(query: str, results: List[Dict[str, str]], sep_str) -> str:
    """Format search results into readable text"""
    if not results:
        return f"No search results found for: {query}"
    
    formatted = f"Search results for '{query}':\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"{i}. **{result['title']}**\n"
        formatted += f"   URL: {result['url']}\n"
        if result['snippet']:
            formatted += f"   Summary: {result['snippet']}\n{sep_str}"
        formatted += "\n"
    
    return formatted

def run(system_prompt: str, initial_query: str, client=None, model: str = None, request_config: Optional[Dict] = None) -> Tuple[str, int]:
    """
    Web search plugin that uses Chrome to search Google and return results
    
    Args:
        system_prompt: System prompt for the conversation
        initial_query: User's query that may contain search requests
        client: OpenAI client (unused for this plugin)
        model: Model name (unused for this plugin) 
        request_config: Optional configuration dict with keys:
            - num_results: Number of search results (default: 10)
            - delay_seconds: Delay between searches in seconds (default: random 4-32)
                            Set to 0 to disable delays, or specify exact seconds
            - headless: Run browser in headless mode (default: False)
            - timeout: Browser timeout in seconds (default: 30)
            - session_manager: BrowserSessionManager instance for session reuse
    
    Returns:
        Tuple of (enhanced_query_with_search_results, completion_tokens)
    """
    # Parse configuration
    config = request_config or {}
    num_results = config.get("num_results", 10)
    delay_seconds = config.get("delay_seconds", None)  # None means random 4-32
    # headless = config.get("headless", False)  # Default to non-headless
    # timeout = config.get("timeout", 30)  # Standard timeout
    session_manager = config.get("session_manager", None)  # For session reuse
    
    sep_str = '#SEP#'

    # Extract search queries from the input
    search_queries = extract_search_queries(initial_query)
    
    if not search_queries:
        return initial_query, 0
    
    # Determine if we should manage the browser lifecycle
    # own_session = session_manager is None
    # print("performing search")
    try:
        # Use provided session manager or create temporary one
        # if own_session:
        #     # Create temporary searcher for standalone use
        #     searcher = GoogleSearcher(headless=headless, timeout=timeout)
        
        enhanced_query = initial_query

        print("============================ search_queries =====================")
        print("search query: \n", search_queries)
        print("num_results: ", num_results)
        
        for query in search_queries:

            # Perform the search
            #if session_manager:
            #    # Use session manager's search method
            #    results = session_manager.search(query, num_results=num_results, delay_seconds=delay_seconds)
            #else:
            #    print("============================ start google search =====================")
            #    # Use temporary searcher
            #    results = searcher.search(query, num_results=num_results, delay_seconds=delay_seconds)
            
            results = tavily_web_search(query, num_results=num_results, delay_seconds=delay_seconds)

            # print(results)
            # Format results
            if results:
                # formatted_results = format_search_results(query, results)
                ### LMCache
                formatted_results = format_search_results_lmc(query, results, sep_str='#SEP#')
                # Append results to the query
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\n{sep_str}{formatted_results}"
            else:
                # No results found - add a note
                enhanced_query = f"{enhanced_query}\n\n[Web Search Results]:\nNo results found for '{query}'. This may be due to network issues or search restrictions."

        print("search result: \n", enhanced_query)
        
        return enhanced_query, 0
        
    except Exception as e:
        error_msg = f"Web search error: {str(e)}"
        enhanced_query = f"{initial_query}\n\n[Web Search Error]: {error_msg}"
        return enhanced_query, 0
        
    finally:
        # Only close if we created our own searcher
        # if own_session and 'searcher' in locals():
        #     searcher.close()
        pass
