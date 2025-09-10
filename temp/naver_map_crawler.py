"""
네이버 맵 리뷰 크롤러 클래스
파일명: naver_map_crawler.py
"""

import requests
import time
import json
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup
import pandas as pd

class NaverMapReviewCrawler:
    def __init__(self, headless=True):
        """
        네이버 맵 리뷰 크롤러 초기화
        
        Args:
            headless (bool): 브라우저를 숨김 모드로 실행할지 여부
        """
        self.setup_driver(headless)
        
    def setup_driver(self, headless):
        """Chrome 드라이버 설정"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.wait = WebDriverWait(self.driver, 10)
        
    def search_place(self, place_name):
        """
        네이버 맵에서 장소 검색
        
        Args:
            place_name (str): 검색할 가게 이름
            
        Returns:
            str: 가게 페이지 URL 또는 None
        """
        try:
            # 네이버 맵 접속
            search_url = f"https://map.naver.com/v5/search/{place_name}"
            print(f"검색 중: {place_name}")
            self.driver.get(search_url)
            
            # 검색 결과 로딩 대기
            time.sleep(3)
            
            # 첫 번째 검색 결과 클릭
            try:
                # 검색 결과 목록에서 첫 번째 항목 찾기
                first_result = self.wait.until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, ".place_bluelink"))
                )
                first_result.click()
                time.sleep(2)
                
                # 상세 페이지로 이동되었는지 확인
                current_url = self.driver.current_url
                if "place" in current_url:
                    print(f"가게 페이지 찾음: {current_url}")
                    return current_url
                    
            except TimeoutException:
                print("검색 결과를 찾을 수 없습니다.")
                return None
                
        except Exception as e:
            print(f"검색 중 오류 발생: {e}")
            return None
            
    def get_place_info(self):
        """현재 페이지에서 가게 기본 정보 추출"""
        try:
            # 가게 이름
            name_element = self.driver.find_element(By.CSS_SELECTOR, ".GHAhO")
            place_name = name_element.text if name_element else "정보 없음"
            
            # 평점
            try:
                rating_element = self.driver.find_element(By.CSS_SELECTOR, ".PXMot em")
                rating = rating_element.text if rating_element else "정보 없음"
            except:
                rating = "정보 없음"
            
            # 주소
            try:
                address_element = self.driver.find_element(By.CSS_SELECTOR, ".LDgIH")
                address = address_element.text if address_element else "정보 없음"
            except:
                address = "정보 없음"
            
            return {
                "name": place_name,
                "rating": rating,
                "address": address
            }
            
        except Exception as e:
            print(f"가게 정보 추출 중 오류: {e}")
            return {"name": "정보 없음", "rating": "정보 없음", "address": "정보 없음"}
    
    def scroll_reviews(self, max_reviews=50):
        """리뷰 영역을 스크롤하여 더 많은 리뷰 로드"""
        try:
            # 리뷰 탭 클릭
            review_tab = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), '리뷰')]"))
            )
            review_tab.click()
            time.sleep(2)
            
            # 리뷰 컨테이너 찾기
            review_container = self.driver.find_element(By.CSS_SELECTOR, ".place_section_content")
            
            # 스크롤하여 리뷰 더 로드
            last_height = self.driver.execute_script("return arguments[0].scrollHeight", review_container)
            loaded_reviews = 0
            
            while loaded_reviews < max_reviews:
                # 컨테이너 내에서 스크롤
                self.driver.execute_script(
                    "arguments[0].scrollTop = arguments[0].scrollHeight", 
                    review_container
                )
                time.sleep(2)
                
                # 새로운 높이 확인
                new_height = self.driver.execute_script("return arguments[0].scrollHeight", review_container)
                
                # 현재 로드된 리뷰 수 확인
                review_elements = self.driver.find_elements(By.CSS_SELECTOR, ".place_bluelink.TYaxT")
                loaded_reviews = len(review_elements)
                
                print(f"로드된 리뷰 수: {loaded_reviews}")
                
                # 더 이상 로드할 리뷰가 없으면 중단
                if new_height == last_height:
                    print("더 이상 로드할 리뷰가 없습니다.")
                    break
                    
                last_height = new_height
                
        except Exception as e:
            print(f"리뷰 스크롤 중 오류: {e}")
    
    def extract_reviews(self):
        """현재 페이지에서 리뷰 데이터 추출"""
        reviews = []
        try:
            # 리뷰 요소들 찾기 - 여러 선택자 시도
            review_selectors = [
                ".place_bluelink.TYaxT",
                ".zPfVt",
                "[class*='review']",
                ".YeINN"
            ]
            
            review_elements = []
            for selector in review_selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    review_elements = elements
                    print(f"리뷰 요소 찾음: {selector}, 개수: {len(elements)}")
                    break
            
            if not review_elements:
                print("리뷰 요소를 찾을 수 없습니다.")
                return reviews
            
            for idx, element in enumerate(review_elements[:50]):  # 최대 50개
                try:
                    # 각 리뷰 클릭하여 상세 내용 보기
                    self.driver.execute_script("arguments[0].click();", element)
                    time.sleep(1)
                    
                    # 리뷰 데이터 추출
                    review_data = self.extract_single_review()
                    if review_data:
                        reviews.append(review_data)
                        print(f"리뷰 {idx+1} 추출 완료")
                    
                    # 뒤로 가기 또는 모달 닫기
                    try:
                        close_btn = self.driver.find_element(By.CSS_SELECTOR, ".place_bluelink")
                        close_btn.click()
                        time.sleep(0.5)
                    except:
                        pass
                        
                except Exception as e:
                    print(f"개별 리뷰 추출 중 오류: {e}")
                    continue
            
            print(f"총 {len(reviews)}개의 리뷰를 추출했습니다.")
            return reviews
            
        except Exception as e:
            print(f"리뷰 추출 중 오류 발생: {e}")
            return reviews
    
    def extract_single_review(self):
        """개별 리뷰 데이터 추출"""
        try:
            # 현재 페이지의 HTML 가져오기
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            
            # 리뷰 텍스트 추출 시도
            review_text = ""
            text_selectors = [
                ".zPfVt",
                "[class*='review_text']",
                ".place_bluelink",
                ".YeINN"
            ]
            
            for selector in text_selectors:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    review_text = element.get_text().strip()
                    break
            
            # 평점 추출 시도
            rating = ""
            rating_selectors = [
                ".PXMot",
                "[class*='rating']",
                ".star_grade"
            ]
            
            for selector in rating_selectors:
                element = soup.select_one(selector)
                if element:
                    rating = element.get_text().strip()
                    break
            
            # 작성자 정보 추출 시도
            author = ""
            author_selectors = [
                ".place_bluelink",
                "[class*='author']",
                ".review_author"
            ]
            
            for selector in author_selectors:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    author = element.get_text().strip()
                    break
            
            if review_text or rating:
                return {
                    "review_text": review_text,
                    "rating": rating,
                    "author": author,
                    "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            
        except Exception as e:
            print(f"개별 리뷰 데이터 추출 중 오류: {e}")
            
        return None
    
    def crawl_reviews(self, place_name, max_reviews=50):
        """
        메인 크롤링 함수
        
        Args:
            place_name (str): 검색할 가게 이름
            max_reviews (int): 수집할 최대 리뷰 수
            
        Returns:
            dict: 가게 정보와 리뷰 데이터
        """
        try:
            # 장소 검색
            place_url = self.search_place(place_name)
            if not place_url:
                return {"error": "장소를 찾을 수 없습니다."}
            
            # 가게 기본 정보 추출
            place_info = self.get_place_info()
            
            # 리뷰 스크롤 및 로드
            self.scroll_reviews(max_reviews)
            
            # 리뷰 데이터 추출
            reviews = self.extract_reviews()
            
            result = {
                "place_info": place_info,
                "reviews": reviews,
                "total_reviews": len(reviews),
                "crawled_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            return {"error": f"크롤링 중 오류 발생: {e}"}
    
    def save_to_csv(self, data, filename):
        """결과를 CSV 파일로 저장"""
        if "error" in data:
            print(f"저장 실패: {data['error']}")
            return
            
        try:
            # 리뷰 데이터를 DataFrame으로 변환
            reviews_df = pd.DataFrame(data["reviews"])
            
            # 가게 정보 추가
            for key, value in data["place_info"].items():
                reviews_df[f"place_{key}"] = value
                
            reviews_df["crawled_at"] = data["crawled_at"]
            
            # CSV 파일로 저장
            reviews_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"결과가 {filename}에 저장되었습니다.")
            
        except Exception as e:
            print(f"파일 저장 중 오류: {e}")
    
    def close(self):
        """드라이버 종료"""
        if self.driver:
            self.driver.quit()