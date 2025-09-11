from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import pymysql
from decimal import Decimal
import datetime
import sys
import os

# googlemaps crawler import
sys.path.append('/Users/ahnsaeyeon/guide-on/ai/temp')
from googlemaps import GoogleMapsCrawler

# sentiment analysis import
import subprocess
import json


app = FastAPI()

class BenchmarkCreditScorer:
    def __init__(self):
        """벤치마크 기반 신용점수 계산기"""
        
        # 요인별 가중치 (기존과 동일)
        self.factor_weights = {
            'Factor_1': {
                'operating_expenses': 0.074535,
                'cost_of_goods_sold': 0.07436,
                'weekend_sales_amount': 0.074554,
                'total_sales_amount': 0.07453,
                'other_expenses': 0.073679,
                'total_salary': 0.0736,
                'rent_expense': 0.072611,
                'weekday_sales_amount': 0.072501,
                'avg_account_balance': 0.07072,
                'operating_profit': 0.069715,
                'weekday_transaction_count': 0.066501,
                'transaction_count': 0.063799,
                'weekend_transaction_count': 0.057164,
                'water_usage_ton': 0.025919,
                'electricity_bill_amount': 0.027752,
                'electricity_usage_kwh': 0.028061
            },
            'Factor_2': {
                'revisit_customer_sales_ratio': 0.083268,
                'sales_cv': 0.084229,
                'new_customer_ratio': 0.078371,
                'operating_profit_ratio': 0.093389,
                'operating_expense_ratio': 0.093389,
                'energy_eff_appliance_ratio': 0.04497,
                'cogs_ratio': 0.049744,
                'customer_review_avg_rating': 0.056357,
                'rent_payment_compliance_rate': 0.025489,
                'card_payment_ratio': 0.057726,
                'cash_payment_ratio': 0.057712,
                'salary_payment_regularity': 0.022855,
                'customer_review_positive_ratio': 0.034197,
                'min_balance_maintenance_ratio': 0.021572,
                'weighted_avg_cash_period': 0.056587,
                'cash_payment_ratio_detail': 0.058045,
                'card_payment_ratio_detail': 0.057432,
                'tax_payment_integrity': 0.012474,
                'cashflow_cv': 0.012193
            },
            'Factor_3': {
                'weekend_avg_transaction_value': 0.229004,
                'weekday_avg_transaction_value': 0.229004,
                'avg_transaction_value': 0.229004,
                'dinner_sales_ratio': 0.131089,
                'lunch_sales_ratio': 0.087794,
                'food_waste_kg_per_day': 0.094106
            },
            'Factor_4': {
                'electricity_usage_kwh': 0.191075,
                'food_waste_kg_per_day': 0.187428,
                'electricity_bill_amount': 0.185198,
                'water_usage_ton': 0.163306,
                'recycle_waste_kg_per_day': 0.160769,
                'gas_usage_m3': 0.112225
            },
            'Factor_5': {
                'cash_payment_ratio': 0.364838,
                'card_payment_ratio': 0.364834,
                'lunch_sales_ratio': 0.270328
            },
            'Factor_6': {
                'card_payment_ratio_detail': 0.341636,
                'cash_payment_ratio_detail': 0.339214,
                'weighted_avg_cash_period': 0.31915
            },
            'Factor_7': {
                'yellow_umbrella_member': 0.484789,
                'yellow_umbrella_amount': 0.362795,
                'yellow_umbrella_months': 0.152416
            },
            'Factor_8': {
                'operating_profit_ratio': 0.435107,
                'operating_expense_ratio': 0.435107,
                'salary_ratio': 0.129785
            }
        }
        
        # 요인별 중요도
        self.factor_importance = {
            'Factor_1': 0.20, 'Factor_2': 0.25,
            'Factor_3': 0.25, 'Factor_4': 0.10,
            'Factor_5': 0.05, 'Factor_6': 0.10,
            'Factor_7': 0.03, 'Factor_8': 0.02
        }
        
        # 역방향 지표
        self.reverse_indicators = {
            'sales_cv', 'cashflow_cv', 'cogs_ratio', 'rent_ratio', 'operating_expenses',
            'operating_expense_ratio', 'weighted_avg_cash_period', 'electricity_usage_kwh',
            'electricity_bill_amount', 'rent_expense', 'gas_usage_m3', 'water_usage_ton',
            'food_waste_kg_per_day', 'other_expenses', 'salary_ratio', 'other_payment_ratio',
            'excessive_withdrawal_frequency', 'origin_price_violation_count'
        }
        
        # 등급 매핑
        self.grade_mapping = [
            (900, "AAA"), (800, "AA"), (700, "A"),
            (600, "BBB"), (500, "BB"), (400, "B"),
            (300, "CCC"), (200, "CC"), (0, "C")
        ]
        
        # 업계 벤치마크
        self.industry_benchmarks = {
            'operating_profit_ratio': {'good': 25.0, 'average': 21.2, 'poor': 15.0},
            'operating_expense_ratio': {'good': 75.0, 'average': 78.8, 'poor': 85.0},
            'cogs_ratio': {'good': 42.0, 'average': 46.4, 'poor': 52.0},
            'salary_ratio': {'good': 14.0, 'average': 15.9, 'poor': 18.0},
            'rent_ratio': {'good': 8.0, 'average': 9.3, 'poor': 12.0},
            'other_expenses': {'good': 6.0, 'average': 7.3, 'poor': 10.0},
            'cashflow_cv': {'good': 0.23, 'average': 0.58, 'poor': 0.93},
            'weighted_avg_cash_period': {'good': 1.0, 'average': 1.4, 'poor': 2.5},
            'excessive_withdrawal_frequency': {'good': 1.0, 'average': 2.1, 'poor': 4.0},
            'avg_account_balance': {'good': 15000000, 'average': 9500000, 'poor': 5000000},
            'min_balance_maintenance_ratio': {'good': 80.0, 'average': 65.0, 'poor': 40.0},
            'cash_payment_ratio': {'good': 40.0, 'average': 52.0, 'poor': 70.0},
            'cash_payment_ratio_detail': {'good': 40.0, 'average': 52.0, 'poor': 70.0},
            'card_payment_ratio': {'good': 55.0, 'average': 45.0, 'poor': 25.0},
            'card_payment_ratio_detail': {'good': 55.0, 'average': 45.0, 'poor': 25.0},
            'other_payment_ratio': {'good': 2.0, 'average': 3.0, 'poor': 5.0},
            'rent_payment_compliance_rate': {'good': 90.0, 'average': 78.0, 'poor': 55.0},
            'utility_payment_compliance_rate': {'good': 87.0, 'average': 74.0, 'poor': 50.0},
            'salary_payment_regularity': {'good': 85.0, 'average': 72.0, 'poor': 50.0},
            'tax_payment_integrity': {'good': 90.0, 'average': 81.0, 'poor': 65.0},
            'total_sales_amount': {'good': 15000000, 'average': 10000000, 'poor': 5000000},
            'weekend_sales_amount': {'good': 6000000, 'average': 4000000, 'poor': 2000000},
            'weekday_sales_amount': {'good': 9000000, 'average': 6000000, 'poor': 3000000},
            'transaction_count': {'good': 1500, 'average': 1000, 'poor': 500},
            'weekday_transaction_count': {'good': 1000, 'average': 650, 'poor': 300},
            'weekend_transaction_count': {'good': 500, 'average': 350, 'poor': 200},
            'avg_transaction_value': {'good': 12000, 'average': 10000, 'poor': 8000},
            'weekday_avg_transaction_value': {'good': 12000, 'average': 10000, 'poor': 8000},
            'weekend_avg_transaction_value': {'good': 12000, 'average': 10000, 'poor': 8000},
            'customer_review_avg_rating': {'good': 4.5, 'average': 3.8, 'poor': 3.0},
            'customer_review_positive_ratio': {'good': 0.85, 'average': 0.70, 'poor': 0.50},
            'electricity_usage_kwh': {'good': 800, 'average': 1200, 'poor': 1800},
            'gas_usage_m3': {'good': 100, 'average': 180, 'poor': 300},
            'water_usage_ton': {'good': 20, 'average': 40, 'poor': 80},
            'food_waste_kg_per_day': {'good': 5, 'average': 15, 'poor': 30},
            'recycle_waste_kg_per_day': {'good': 2, 'average': 5, 'poor': 10},
        }
    
    def normalize_with_benchmark(self, variable, value):
        """벤치마크 기반 점수 정규화"""
        # None이나 NaN 체크
        if value is None or pd.isna(value):
            return 50
        
        # Decimal이나 다른 타입을 float로 변환
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 50
        
        # inf 체크
        if np.isinf(value):
            return 50
        
        # 벤치마크가 있는 경우
        if variable in self.industry_benchmarks:
            benchmark = self.industry_benchmarks[variable]
            good_val = benchmark['good']
            avg_val = benchmark['average']
            poor_val = benchmark['poor']
            
            if variable in self.reverse_indicators:
                # 역방향: 값이 작을수록 좋음
                if value <= good_val:
                    score = 85
                elif value >= poor_val:
                    score = 25
                elif value <= avg_val:
                    # good ~ average 구간: 85 ~ 55점
                    ratio = (value - good_val) / (avg_val - good_val)
                    score = 85 - ratio * 30
                else:
                    # average ~ poor 구간: 55 ~ 25점
                    ratio = (value - avg_val) / (poor_val - avg_val)
                    score = 55 - ratio * 30
            else:
                # 정방향: 값이 클수록 좋음
                if value >= good_val:
                    score = 85
                elif value <= poor_val:
                    score = 25
                elif value >= avg_val:
                    # average ~ good 구간: 55 ~ 85점
                    ratio = (value - avg_val) / (good_val - avg_val)
                    score = 55 + ratio * 30
                else:
                    # poor ~ average 구간: 25 ~ 55점
                    ratio = (value - poor_val) / (avg_val - poor_val)
                    score = 25 + ratio * 30
        else:
            # 벤치마크가 없는 경우: 기본 변환
            if value > 0:
                score = min(85, 30 + np.log10(value + 1) * 10)
            else:
                score = 30
            
            if variable in self.reverse_indicators:
                score = 100 - score
        
        return max(15, min(90, score))
    
    def calculate_factor_score(self, factor_name, row_data):
        """요인 점수 계산"""
        factor_vars = self.factor_weights.get(factor_name, {})
        if not factor_vars:
            return 50
        
        weighted_sum, total_weight = 0, 0
        
        for variable, weight in factor_vars.items():
            if variable in row_data:
                score = self.normalize_with_benchmark(variable, row_data[variable])
                weighted_sum += score * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 50
    
    def calculate_single_score(self, row_data):
        """단일 행 신용점수 계산"""
        factor_scores = {
            f: self.calculate_factor_score(f, row_data)
            for f in self.factor_weights.keys()
        }
        
        # 최종 점수 계산
        total_importance = sum(self.factor_importance.values())
        final_score = sum(
            (factor_scores[f] * self.factor_importance[f]) / total_importance
            for f in factor_scores
        )
        
        # 신용점수: 15~90 점수를  350~950점 범위로 변환
        credit_score = int(350 + (final_score - 15) * (950 - 350) / (90 - 15))
        credit_score = max(350, min(950, credit_score))
        credit_grade = self.get_credit_grade(credit_score)
        
        return {
            'store_id': row_data.get('store_id', 0),
            'business_registration_no': row_data.get('business_registration_no', ''),
            '신용점수': credit_score,
            '신용등급': credit_grade,
            '최종점수_raw': round(final_score, 2),
            **{f: round(s, 2) for f, s in factor_scores.items()}
        }
    
    def get_credit_grade(self, score):
        """점수 → 등급"""
        for threshold, grade in self.grade_mapping:
            if score >= threshold:
                return grade
        return "C"

    def clean_db_row(self, row):
        """DB 데이터 정리"""
        cleaned_row = {}
        for key, value in row.items():
            if value is None:
                cleaned_row[key] = None
            elif isinstance(value, (int, float, str)):
                cleaned_row[key] = value
            else:
                # Decimal, datetime 등을 적절히 변환
                try:
                    cleaned_row[key] = float(value)
                except (TypeError, ValueError):
                    cleaned_row[key] = str(value)
        return cleaned_row


# 글로벌 스코어러 인스턴스
scorer = BenchmarkCreditScorer()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pymysql
import pandas as pd
import json
import subprocess
from typing import Optional

# FA_score import (간소화된 함수)
from FA_score import get_credit_score as fa_get_credit_score

app = FastAPI(title="신용점수 계산 API (테이블별 점수 포함)")

class CreditScoreRequest(BaseModel):
    store_id: int
    summary_year_month: str = "2025-08"

class GoogleMapsRequest(BaseModel):
    store_name: str
    target_count: int = 25

class GoogleMapsResponse(BaseModel):
    message: str
    reviews_count: int
    store_name: str

class SentimentAnalysisResponse(BaseModel):
    positive_ratio: float

class CreditScoreResponse(BaseModel):
    store_id: int
    business_registration_no: str
    신용점수: int
    신용등급: str
    최종점수_raw: float
    Factor_1: float
    Factor_2: float
    Factor_3: float
    Factor_4: float
    Factor_5: float
    Factor_6: float
    Factor_7: float
    Factor_8: float

class TableScores(BaseModel):
    """테이블별 점수 모델"""
    score_raw: float
    score_scaled: int
    data_count: int

class DetailedTableScores(BaseModel):
    """상세한 테이블별 점수 정보"""
    sales_summary: TableScores
    financial_info: TableScores
    operational_info: TableScores

class HybridCreditScoreResponse(BaseModel):
    store_id: int
    credit_score: int
    final_score: float
    total_sales_amount: float
    operating_profit_ratio: float
    mom_growth_rate: float
    tax_payment_integrity: float
    cashflow_cv: float
    # 테이블별 점수 추가
    sales_summary_score_scaled: int
    financial_info_score_scaled: int
    operational_info_score_scaled: int
    message: str

class EnhancedCreditScoreResponse(BaseModel):
    """테이블별 점수가 포함된 향상된 신용점수 응답"""
    store_id: int
    credit_score: int
    final_score: float
    table_scores: DetailedTableScores
    key_metrics: dict
    message: str

def get_db_connection():
    """DB 연결"""
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='Ab1515!!',
        db='guideon',
        charset='utf8'
    )

def get_store_data_from_db(store_id):
    """
    MySQL 데이터베이스에서 store_id에 해당하는 store_summary 데이터를 불러오는 함수
    
    Args:
        store_id (int): 상점 ID
    
    Returns:
        pandas.DataFrame: 해당 store_id의 데이터 (컬럼명 포함)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # store_summary 테이블에서 해당 store_id 데이터 조회
        query = "SELECT * FROM store_summary WHERE session_id = %s"
        cursor.execute(query, (store_id,))
        
        # 결과 가져오기
        result = cursor.fetchall()
        
        if not result:
            return None
        
        # 컬럼명 가져오기
        columns = [desc[0] for desc in cursor.description]
        
        # DataFrame으로 변환
        df = pd.DataFrame(list(result), columns=columns)
        
        cursor.close()
        conn.close()
        
        return df
        
    except Exception as e:
        print(f"데이터베이스 연결 오류: {e}")
        return None

@app.get("/")
async def root():
    return {"message": "신용점수 계산 API (테이블별 점수 포함)"}

# @app.post("/hybrid-credit-score-result", response_model=CreditScoreResponse)
# async def calculate_credit_score(request: CreditScoreRequest):
#     """기존 API 유지 (하위 호환성)"""
#     try:
#         # DB 연결
#         conn = get_db_connection()
#         cursor = conn.cursor(pymysql.cursors.DictCursor)
        
#         # 데이터 조회
#         query = """
#         SELECT * FROM store_unified_summary 
#         WHERE store_id = %s AND summary_year_month = %s
#         """
#         cursor.execute(query, (request.store_id, request.summary_year_month))
#         row = cursor.fetchone()
        
#         if not row:
#             raise HTTPException(
#                 status_code=404, 
#                 detail=f"Store ID {request.store_id}의 {request.summary_year_month} 데이터를 찾을 수 없습니다."
#             )
        
#         # 데이터 정리 및 신용점수 계산
#         cleaned_row = scorer.clean_db_row(row)
#         result = scorer.calculate_single_score(cleaned_row)
        
#         # DB 연결 종료
#         cursor.close()
#         conn.close()
        
#         return CreditScoreResponse(**result)
        
#     except pymysql.Error as e:
#         raise HTTPException(status_code=500, detail=f"데이터베이스 오류: {str(e)}")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"계산 오류: {str(e)}")

# @app.get("/hybrid-credit-score-result/{store_id}")
# async def get_credit_score(store_id: int, summary_year_month: str = "2025-08"):
#     """GET 방식으로도 조회 가능 (기존 API 유지)"""
#     request = CreditScoreRequest(store_id=store_id, summary_year_month=summary_year_month)
#     return await calculate_credit_score(request)

@app.post("/hybrid-credit-score/{session_id}")
async def save_hybrid_credit_score(session_id: int):
    """
    신용점수를 계산하고 member_credit 테이블에 저장하는 API
    
    Args:
        session_id (int): 세션 ID (path variable)
    
    Returns:
        dict: 저장 완료 메시지
    """
    try:
        # 1. 데이터베이스에서 데이터 조회
        store_data = get_store_data_from_db(session_id)
        
        if store_data is None or store_data.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"Session ID {session_id}에 해당하는 데이터를 찾을 수 없습니다."
            )
        
        # 2. 신용점수 계산 (FA_score.py의 result_df 사용)   
        result_df = fa_get_credit_score(store_data)
        
        if result_df is None or result_df.empty:
            raise HTTPException(
                status_code=500,
                detail="신용점수 계산 중 오류가 발생했습니다."
            )
        
        # 3. 필요한 값만 추출
        result_row = result_df.iloc[0]
        credit_score = int(result_row['credit_score'])
        
        # 4. member_credit 테이블에 모든 점수 저장
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # member_credit 테이블에 모든 점수 저장 (UPDATE)
            save_query = """
            UPDATE member_credit
            SET
                hybrid_credit_score = %s,
                sales_summary_score_scaled = %s,
                financial_info_score_scaled = %s,
                operational_info_score_scaled = %s
            WHERE
                session_id = %s;
            """
            cursor.execute(save_query, (
                credit_score, 
                int(result_row['sales_summary_score_scaled']),
                int(result_row['financial_info_score_scaled']),
                int(result_row['operational_info_score_scaled']),
                session_id
            ))
            conn.commit()
            
            cursor.close()
            conn.close()
            
        except Exception as db_error:
            print(f"DB 저장 오류: {db_error}")
            raise HTTPException(status_code=500, detail=f"데이터베이스 저장 오류: {str(db_error)}")
        
        # 5. API 응답 반환
        return {
            "message": "신용점수 계산 및 저장이 성공적으로 완료되었습니다.",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류가 발생했습니다: {(e)}"
        )

@app.get("/hybrid-credit-score/result/{session_id}")
async def get_hybrid_credit_score_result(session_id: int):
    """
    member_credit 테이블에서 해당 session_id의 신용점수 정보를 조회하는 API
    
    Args:
        session_id (int): 세션 ID (path variable)
    
    Returns:
        dict: 신용점수 정보
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        # member_credit 테이블에서 해당 session_id의 데이터 조회
        query = """
        SELECT hybrid_credit_score, sales_summary_score_scaled, 
               financial_info_score_scaled, operational_info_score_scaled
        FROM member_credit 
        WHERE session_id = %s
        """
        cursor.execute(query, (session_id,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if not result:
            raise HTTPException(
                status_code=404, 
                detail=f"Session ID {session_id}에 해당하는 데이터를 찾을 수 없습니다."
            )
        
        return {
            "session_id": session_id,
            "hybrid_credit_score": result['hybrid_credit_score'],
            "sales_summary_score_scaled": result['sales_summary_score_scaled'],
            "financial_info_score_scaled": result['financial_info_score_scaled'],
            "operational_info_score_scaled": result['operational_info_score_scaled']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"서버 오류가 발생했습니다: {str(e)}"
        )

# @app.get("/enhanced-credit-score/{store_id}")
# async def get_enhanced_credit_score(store_id: int):
#     """
#     간소화된 신용점수 계산 API (hybrid-credit-score와 동일)
#     """
#     return await get_hybrid_credit_score(store_id)

# @app.get("/detailed-credit-analysis/{store_id}")
# async def get_detailed_credit_analysis(store_id: int):
#     """
#     간소화된 상세 신용분석 결과 API
#     """
#     try:
#         # 기본 신용점수 정보 가져오기
#         basic_result = await get_hybrid_credit_score(store_id)
        
#         # 추가 분석 정보 포함
#         credit_score = basic_result['credit_score']
        
#         return {
#             "analysis_timestamp": pd.Timestamp.now().isoformat(),
#             "store_analysis": basic_result,
#             "summary": {
#                 "credit_score": credit_score,
#                 "risk_level": "낮음" if credit_score >= 700 else "보통" if credit_score >= 500 else "높음",
#                 "table_contribution": {
#                     "sales_percentage": round((basic_result['sales_summary_score_scaled'] / credit_score) * 100, 1),
#                     "financial_percentage": round((basic_result['financial_info_score_scaled'] / credit_score) * 100, 1),
#                     "operational_percentage": round((basic_result['operational_info_score_scaled'] / credit_score) * 100, 1)
#                 }
#             }
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"서버 오류가 발생했습니다: {str(e)}"
#         )

@app.post("/search/{store_name}", response_model=GoogleMapsResponse)
async def crawl_googlemaps_reviews(store_name: str, target_count: int = 10):
    """Google Maps 리뷰 크롤링 API"""
    try:
        # GoogleMapsCrawler 인스턴스 생성
        crawler = GoogleMapsCrawler(output_dir="/Users/ahnsaeyeon/guide-on/ai/temp")
        
        # 리뷰 크롤링 실행
        crawler.scrape_reviews(search_query=store_name, target_count=target_count)
        
        # 데이터베이스에 저장
        crawler.save_to_database()
        
        # 브라우저 종료
        if hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()
        
        return GoogleMapsResponse(
            message=f"{store_name} 리뷰 크롤링이 완료되었습니다.",
            reviews_count=len(crawler.review_data),
            store_name=store_name
        )
        
    except Exception as e:
        # 에러 발생 시 브라우저 정리
        if 'crawler' in locals() and hasattr(crawler, 'driver') and crawler.driver:
            crawler.driver.quit()
        raise HTTPException(status_code=500, detail=f"크롤링 오류: {str(e)}")

@app.post("/sentimental-analysis-result/{session_id}", response_model=SentimentAnalysisResponse)
async def get_sentimental_analysis_result(session_id: int):
    """감정 분석 결과를 계산하고 store_summary 테이블에 저장하는 API"""
    try:
        # sentimental_analysis.py 실행하여 positive_ratio와 rating_average 얻기
        result = subprocess.run(
            ["python", "/Users/ahnsaeyeon/guide-on/ai/temp/sentimental_analysis.py"],
            capture_output=True,
            text=True,
            cwd="/Users/ahnsaeyeon/guide-on/ai/temp"
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"감정 분석 실행 오류: {result.stderr}")
        
        # JSON 출력에서 positive_ratio와 rating_average 값 추출
        output_lines = result.stdout.strip().split('\n')
        json_line = output_lines[-1]  # 마지막 줄이 JSON 결과
        
        try:
            result_data = json.loads(json_line)
            positive_ratio = result_data["positive_ratio"] * 100
            rating_average = result_data["rating_average"]
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(status_code=500, detail=f"결과 파싱 오류: {str(e)}")
        
        # store_summary 테이블에 데이터 저장
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # store_summary 테이블에 customer_review_positive_ratio와 customer_review_avg_rating 저장
            update_query = """
            UPDATE store_summary 
            SET customer_review_positive_ratio = %s, customer_review_avg_rating = %s
            WHERE session_id = %s
            """
            cursor.execute(update_query, (positive_ratio, rating_average, session_id))
            conn.commit()
            
            cursor.close()
            conn.close()
            
        except Exception as db_error:
            print(f"DB 저장 오류: {db_error}")
            # DB 저장 실패해도 API 응답은 정상 반환
        
        return SentimentAnalysisResponse(positive_ratio=positive_ratio)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"감정 분석 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)