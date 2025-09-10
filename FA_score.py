#!/usr/bin/env python3
"""
신용평가 시스템 - 상점 신용점수 계산 파이프라인 (테이블별 점수 포함)
"""

import pandas as pd
import numpy as np
import joblib


def get_credit_score(new_data_df):
    """
    신규 데이터(DataFrame)를 입력받아 최종 신용 점수를 계산하는 파이프라인 함수.
    
    [필수 조건]
    1. 이 함수와 같은 경로에 'credit_scoring_artifacts.pkl' 파일이 있어야 합니다.
    2. new_data_df는 학습에 사용된 초기 데이터와 동일한 컬럼명을 가지고 있어야 합니다.
    
    :param new_data_df: 신규 상점 데이터 (1개 또는 여러 개)
    :return: 최종 점수가 포함된 DataFrame
    """
    try:
        # 1. 저장된 모델 자산(artifacts) 불러오기
        artifacts = joblib.load('credit_scoring_artifacts.pkl')
    except FileNotFoundError:
        print("에러: 'credit_scoring_artifacts.pkl' 파일을 찾을 수 없습니다.")
        print("학습 코드를 먼저 실행하여 모델 자산 파일을 생성해야 합니다.")
        return None

    # --- 2. 전처리 단계 ---
    # 저장된 객체 및 컬럼 목록 추출
    scaler = artifacts['scaler']
    scaler_cols = artifacts['scaler_columns']
    selector = artifacts.get('selector')  # selector가 없을 수도 있으므로 .get() 사용
    to_drop_corr = artifacts['to_drop_corr']
    fa_cols = artifacts['fa_columns']

    # 2-1. Scaler가 학습했던 컬럼들만 신규 데이터에서 선택
    # 만약 신규 데이터에 컬럼이 부족하면 0으로 채움
    processed_df = pd.DataFrame(columns=scaler_cols, index=new_data_df.index)
    common_cols = [col for col in scaler_cols if col in new_data_df.columns]
    processed_df[common_cols] = new_data_df[common_cols]
    processed_df.fillna(0, inplace=True)

    # 2-2. 저장된 Scaler로 표준화 수행 (.transform)
    scaled_data = scaler.transform(processed_df)
    scaled_df = pd.DataFrame(scaled_data, columns=scaler_cols, index=processed_df.index)

    # 2-3. 저장된 VarianceThreshold로 변수 제거 (.transform)
    if selector:
        var_filtered_data = selector.transform(scaled_df)
        filtered_df = pd.DataFrame(var_filtered_data, 
                                   columns=scaled_df.columns[selector.get_support()],
                                   index=scaled_df.index)
    else:
        filtered_df = scaled_df

    # 2-4. 저장된 상관관계 변수 목록으로 변수 제거 (.drop)
    filtered_df = filtered_df.drop(columns=to_drop_corr, errors='ignore')

    # 2-5. 최종적으로 FactorAnalyzer가 학습했던 변수/순서와 동일하게 맞춤
    final_processed_data = filtered_df[fa_cols]
    
    # --- 3. 최종 점수 계산 단계 (테이블별 점수 포함) ---
    final_scores_df = pd.DataFrame(index=final_processed_data.index)
    final_scores_df['final_score'] = 0
    
    # 테이블별 점수 초기화
    final_scores_df['sales_summary_score'] = 0
    final_scores_df['financial_info_score'] = 0
    final_scores_df['operational_info_score'] = 0
    
    fa_model = artifacts['factor_analyzer']
    loadings = artifacts['loadings']
    mapping = artifacts['variable_factor_mapping']
    weights = artifacts['factor_weights']


    # 테이블별 변수 정의
    sales_summary_vars = [
        'total_sales_amount', 'weekday_sales_amount', 'weekend_sales_amount',
        'lunch_sales_ratio', 'dinner_sales_ratio', 'transaction_count', 'weekday_transaction_count',
        'weekend_transaction_count', 'mom_growth_rate', 'yoy_growth_rate', 'sales_cv',
        'avg_transaction_value',
        'cash_payment_ratio', 'revisit_customer_sales_ratio', 'new_customer_ratio'
    ]
    
    financial_vars = [
        'operating_profit', 'cost_of_goods_sold', 'total_salary', 'operating_expenses',
        'rent_expense', 'other_expenses', 'operating_profit_ratio', 'cogs_ratio',
        'salary_ratio', 'rent_ratio', 'operating_expense_ratio', 'cash_payment_ratio_detail',
        'card_payment_ratio_detail', 'other_payment_ratio', 'weighted_avg_cash_period',
        'cashflow_cv', 'avg_account_balance', 'min_balance_maintenance_ratio',
        'excessive_withdrawal_frequency', 'rent_payment_compliance_rate',
        'utility_payment_compliance_rate', 'salary_payment_regularity', 'tax_payment_integrity'
    ]
    
    operational_vars = [
        'electricity_usage_kwh', 'electricity_bill_amount', 'gas_usage_m3', 'water_usage_ton',
        'energy_eff_appliance_ratio', 'participate_energy_eff_support', 'participate_high_eff_equip_support',
        'food_waste_kg_per_day', 'recycle_waste_kg_per_day', 'yellow_umbrella_member',
        'yellow_umbrella_months', 'yellow_umbrella_amount', 'employment_insurance_employees',
        'customer_review_avg_rating', 'customer_review_positive_ratio', 'hygiene_certified',
        'origin_price_violation_count'
    ]

    for i in range(fa_model.n_factors):
        variables_in_factor = mapping[mapping == i].index
        factor_score_col = pd.Series(0.0, index=final_processed_data.index)
        
        # 각 변수별로 기여도 계산
        sales_contribution = pd.Series(0.0, index=final_processed_data.index)
        financial_contribution = pd.Series(0.0, index=final_processed_data.index)
        operational_contribution = pd.Series(0.0, index=final_processed_data.index)
        
        for var in variables_in_factor:
            if var in final_processed_data.columns:
                var_contribution = final_processed_data[var] * loadings.loc[var, i] * weights[i]
                factor_score_col += final_processed_data[var] * loadings.loc[var, i]
                
                # 변수가 어느 테이블에 속하는지 확인하고 해당 테이블 점수에 추가
                if var in sales_summary_vars:
                    sales_contribution += var_contribution
                elif var in financial_vars:
                    financial_contribution += var_contribution
                elif var in operational_vars:
                    operational_contribution += var_contribution
        
        # 각 테이블별 점수 누적
        final_scores_df['sales_summary_score'] += sales_contribution
        final_scores_df['financial_info_score'] += financial_contribution
        final_scores_df['operational_info_score'] += operational_contribution
        
        # 전체 점수 계산
        final_scores_df['final_score'] += factor_score_col * weights[i]
        
    # --- 4. 0-1000점 척도 변환 단계 ---
    lower_b, upper_b = artifacts['score_lower_bound'], artifacts['score_upper_bound']
    
    # 전체 신용점수 변환
    clipped_score = final_scores_df['final_score'].clip(lower_b, upper_b)
    final_scores_df['credit_score'] = ((clipped_score - lower_b) / (upper_b - lower_b)) * 1000
    final_scores_df['credit_score'] = final_scores_df['credit_score'].fillna(0).astype(int)
    
    # 테이블별 점수도 동일한 방식으로 0-1000 척도로 변환
    # sales_summary_score 변환
    clipped_sales = final_scores_df['sales_summary_score'].clip(lower_b, upper_b)
    final_scores_df['sales_summary_score_scaled'] = ((clipped_sales - lower_b) / (upper_b - lower_b)) * 1000
    final_scores_df['sales_summary_score_scaled'] = final_scores_df['sales_summary_score_scaled'].fillna(0).astype(int)
    
    # financial_info_score 변환
    clipped_financial = final_scores_df['financial_info_score'].clip(lower_b, upper_b)
    final_scores_df['financial_info_score_scaled'] = ((clipped_financial - lower_b) / (upper_b - lower_b)) * 1000
    final_scores_df['financial_info_score_scaled'] = final_scores_df['financial_info_score_scaled'].fillna(0).astype(int)
    
    # operational_info_score 변환
    clipped_operational = final_scores_df['operational_info_score'].clip(lower_b, upper_b)
    final_scores_df['operational_info_score_scaled'] = ((clipped_operational - lower_b) / (upper_b - lower_b)) * 1000
    final_scores_df['operational_info_score_scaled'] = final_scores_df['operational_info_score_scaled'].fillna(0).astype(int)
    
    # 원본 데이터와 최종 점수 합쳐서 반환
    return pd.concat([new_data_df, final_scores_df], axis=1)


