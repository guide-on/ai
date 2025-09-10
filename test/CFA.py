import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bartlett
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import warnings
warnings.filterwarnings('ignore')

class WeightCalculator:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.factor_results = {}
        self.weights = {}
        
        # 컬럼 그룹 정의
        self.column_groups = {
            '매출_안정성_성장성': [
                'total_sales_amount', 'weekday_sales_amount', 'weekend_sales_amount',
                'lunch_sales_ratio', 'dinner_sales_ratio', 'transaction_count',
                'weekday_transaction_count', 'weekend_transaction_count',
                'mom_growth_rate', 'yoy_growth_rate', 'sales_cv',
                'avg_transaction_value', 'weekday_avg_transaction_value', 'weekend_avg_transaction_value',
                'cash_payment_ratio', 'card_payment_ratio', 'revisit_customer_sales_ratio', 'new_customer_ratio'
            ],
            '현금흐름_건전성': [
                'operating_profit', 'cost_of_goods_sold', 'total_salary', 'operating_expenses',
                'rent_expense', 'other_expenses', 'operating_profit_ratio', 'cogs_ratio',
                'salary_ratio', 'rent_ratio', 'operating_expense_ratio',
                'cash_payment_ratio_detail', 'card_payment_ratio_detail', 'other_payment_ratio',
                'weighted_avg_cash_period', 'cashflow_cv', 'avg_account_balance',
                'min_balance_maintenance_ratio', 'excessive_withdrawal_frequency',
                'rent_payment_compliance_rate', 'utility_payment_compliance_rate',
                'salary_payment_regularity', 'tax_payment_integrity'
            ],
            'ESG': [
                'electricity_usage_kwh', 'electricity_bill_amount', 'gas_usage_m3', 'water_usage_ton',
                'energy_eff_appliance_ratio', 'participate_energy_eff_support', 
                'participate_high_eff_equip_support', 'food_waste_kg_per_day', 'recycle_waste_kg_per_day',
                'yellow_umbrella_member', 'yellow_umbrella_months', 'yellow_umbrella_amount',
                'employment_insurance_employees', 'customer_review_avg_rating', 
                'customer_review_positive_ratio', 'hygiene_certified', 'origin_price_violation_count'
            ]
        }
        
        # 역방향 지표 (낮을수록 좋은 지표)
        self.reverse_indicators = [
            'sales_cv', 'cashflow_cv', 'cogs_ratio', 'rent_ratio', 'operating_expense_ratio',
            'weighted_avg_cash_period', 'excessive_withdrawal_frequency',
            'electricity_usage_kwh', 'electricity_bill_amount', 'gas_usage_m3', 
            'water_usage_ton', 'food_waste_kg_per_day', 'origin_price_violation_count'
        ]
    
    def load_and_preprocess_data(self, csv_file_path):
        """CSV 파일 로드 및 전처리"""
        print(f"데이터 로딩 중: {csv_file_path}")
        
        # CSV 파일 읽기
        try:
            df = pd.read_csv("/Users/ahnsaeyeon/guide-on/ai/store_unified_summary.csv", encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv("/Users/ahnsaeyeon/guide-on/ai/store_unified_summary.csv", encoding='cp949')

        print(f"원본 데이터 크기: {df.shape}")
        print(f"컬럼 수: {len(df.columns)}")
        
        # 사용 가능한 컬럼만 추출
        available_columns = {}
        for group_name, columns in self.column_groups.items():
            available_cols = [col for col in columns if col in df.columns]
            available_columns[group_name] = available_cols
            print(f"\n{group_name} 그룹:")
            print(f"  전체 컬럼: {len(columns)}개")
            print(f"  사용가능: {len(available_cols)}개")
            if len(available_cols) != len(columns):
                missing_cols = [col for col in columns if col not in df.columns]
                print(f"  누락된 컬럼: {missing_cols}")
        
        self.available_columns = available_columns
        
        # 전처리
        df_processed = self._preprocess_data(df)
        
        return df_processed
    
    def _preprocess_data(self, df):
        """데이터 전처리"""
        df_processed = df.copy()
        
        print("데이터 전처리 중...")
        
        # 1. 모든 NaN과 무한값을 0으로 대체
        print("1. NaN과 무한값을 0으로 대체 중...")
        df_processed = df_processed.replace([np.inf, -np.inf, np.nan], 0)
        
        # 2. Boolean 컬럼을 0/1로 변환
        print("2. Boolean 컬럼 변환 중...")
        boolean_cols = ['yellow_umbrella_member', 'participate_energy_eff_support', 
                       'participate_high_eff_equip_support', 'hygiene_certified']
        for col in boolean_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # 3. 역방향 지표 변환 (낮을수록 좋은 지표를 높을수록 좋게 변환)
        print("3. 역방향 지표 변환 중...")
        for col in self.reverse_indicators:
            if col in df_processed.columns:
                max_val = df_processed[col].max()
                min_val = df_processed[col].min()
                if max_val != min_val:
                    df_processed[col] = max_val - df_processed[col] + min_val
        
        print(f"\n전처리 완료. 최종 데이터 크기: {df_processed.shape}")
        return df_processed
    
    def check_factor_analysis_assumptions(self, data, group_name):
        """요인분석 가정 검증"""
        print(f"\n=== {group_name} 요인분석 가정 검증 ===")
        
        # 1. 표본 크기 확인
        n_samples, n_features = data.shape
        print(f"표본 수: {n_samples}, 변수 수: {n_features}")
        print(f"표본/변수 비율: {n_samples/n_features:.2f} (권장: 5 이상)")
        
        # 2. Bartlett's 구형성 검정
        try:
            chi_square_value, p_value = calculate_bartlett_sphericity(data)
            print(f"Bartlett 구형성 검정: chi2={chi_square_value:.2f}, p-value={p_value:.4f}")
            if p_value < 0.05:
                print("  ✓ 구형성 귀무가설 기각 - 요인분석 적합")
            else:
                print("  ✗ 구형성 귀무가설 채택 - 요인분석 부적합")
        except:
            print("  Bartlett 검정 실행 불가")
        
        # 3. KMO 표본적합도 검정
        try:
            kmo_all, kmo_model = calculate_kmo(data)
            print(f"KMO 전체 적합도: {kmo_model:.3f}")
            if kmo_model >= 0.7:
                print("  ✓ 우수한 적합도 (≥0.7)")
            elif kmo_model >= 0.6:
                print("  ○ 적당한 적합도 (0.6-0.7)")
            else:
                print("  ✗ 부적합한 적합도 (<0.6)")
        except:
            print("  KMO 검정 실행 불가")
        
        return True
    
    def perform_factor_analysis(self, df):
        """각 그룹별 요인분석 수행"""
        print("\n" + "="*60)
        print("요인분석 수행 중...")
        print("="*60)
        
        group_results = {}
        
        for group_name, columns in self.available_columns.items():
            if len(columns) < 3:
                print(f"\n{group_name}: 변수가 너무 적음 ({len(columns)}개). 건너뜀.")
                continue
            
            print(f"\n{group_name} 분석 중... ({len(columns)}개 변수)")
            
            # 해당 그룹의 데이터 추출
            group_data = df[columns].copy()
            
            # 결측치 처리
            group_data_filled = pd.DataFrame(
                self.imputer.fit_transform(group_data), 
                columns=columns
            )
            
            # 표준화
            group_data_scaled = self.scaler.fit_transform(group_data_filled)
            
            # 가정 검증
            self.check_factor_analysis_assumptions(group_data_scaled, group_name)
            
            # 최적 요인 수 결정
            optimal_factors = self._determine_optimal_factors(group_data_scaled, group_name)
            
            # 요인분석 수행
            fa = FactorAnalyzer(n_factors=optimal_factors, rotation='varimax')
            fa.fit(group_data_scaled)
            
            
            # 결과 저장
            loadings = fa.loadings_
            eigenvalues = fa.get_eigenvalues()[0]
            communalities = fa.get_communalities()
            
            group_results[group_name] = {
                'factor_analyzer': fa,
                'loadings': loadings,
                'eigenvalues': eigenvalues,
                'communalities': communalities,
                'columns': columns,
                'n_factors': optimal_factors,
                'explained_variance': np.sum(eigenvalues[:optimal_factors]) / len(columns)
            }
            
            print(f"  최적 요인 수: {optimal_factors}")
            print(f"  설명 분산 비율: {group_results[group_name]['explained_variance']:.3f}")
        
        self.factor_results = group_results
        return group_results
    
    def _determine_optimal_factors(self, data, group_name, max_factors=None):
        """최적 요인 수 결정 (Kaiser 기준과 Scree plot)"""
        n_variables = data.shape[1]
        max_factors = max_factors or min(n_variables, 8)
        
        # 다양한 요인 수로 분석
        eigenvalues_list = []
        for n_factors in range(1, max_factors + 1):
            try:
                fa = FactorAnalyzer(n_factors=n_factors)
                fa.fit(data)
                eigenvalues_list.append(fa.get_eigenvalues()[0])
            except:
                break
        
        if not eigenvalues_list:
            return 1
        
        # Kaiser 기준 (고유값 > 1)
        max_eigenvalues = eigenvalues_list[-1] if eigenvalues_list else np.array([1])
        kaiser_factors = np.sum(max_eigenvalues > 1)
        
        # 최소 1개, 최대 변수 수의 절반
        optimal_factors = max(1, min(kaiser_factors, n_variables // 2))
        
        print(f"  Kaiser 기준 요인 수: {kaiser_factors}")
        print(f"  선택된 요인 수: {optimal_factors}")
        
        return optimal_factors
    
    def calculate_weights(self):
        """가중치 계산"""
        print("\n" + "="*60)
        print("가중치 계산 중...")
        print("="*60)
        
        weights = {}
        
        for group_name, result in self.factor_results.items():
            loadings = result['loadings']
            columns = result['columns']
            
            # 각 변수의 communality (공통성) 기반 가중치
            communalities = result['communalities']
            
            # 로딩값의 절댓값 평균으로 가중치 계산
            variable_weights = np.mean(np.abs(loadings), axis=1)
            
            # communality를 반영한 최종 가중치
            final_weights = variable_weights * communalities
            
            # 정규화 (합이 1이 되도록)
            final_weights = final_weights / np.sum(final_weights)
            
            # 컬럼별 가중치 저장
            group_weights = {}
            for i, col in enumerate(columns):
                group_weights[col] = final_weights[i]
            
            weights[group_name] = group_weights
            
            print(f"\n{group_name} 그룹 가중치:")
            sorted_weights = sorted(group_weights.items(), key=lambda x: x[1], reverse=True)
            for col, weight in sorted_weights[:10]:  # 상위 10개만 출력
                print(f"  {col:<40} {weight:.4f}")
            if len(sorted_weights) > 10:
                print(f"  ... (총 {len(sorted_weights)}개 변수)")
        
        self.weights = weights
        return weights
    
    def save_results(self, output_file='weight_analysis_results.xlsx'):
        """결과를 Excel 파일로 저장"""
        print(f"\n결과 저장 중: {output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # 1. 그룹별 가중치 요약
            summary_data = []
            for group_name, group_weights in self.weights.items():
                for col, weight in group_weights.items():
                    summary_data.append({
                        '그룹': group_name,
                        '변수명': col,
                        '가중치': weight,
                        '순위': 0  # 나중에 계산
                    })
            
            summary_df = pd.DataFrame(summary_data)
            
            # 그룹별 순위 계산
            for group in summary_df['그룹'].unique():
                mask = summary_df['그룹'] == group
                summary_df.loc[mask, '순위'] = summary_df.loc[mask, '가중치'].rank(ascending=False)
            
            summary_df.to_excel(writer, sheet_name='가중치_요약', index=False)
            
            # 2. 그룹별 상세 결과
            for group_name, result in self.factor_results.items():
                # 로딩 매트릭스
                loadings_df = pd.DataFrame(
                    result['loadings'], 
                    index=result['columns'],
                    columns=[f'Factor_{i+1}' for i in range(result['n_factors'])]
                )
                loadings_df['Communality'] = result['communalities']
                loadings_df['Weight'] = [self.weights[group_name][col] for col in result['columns']]
                
                sheet_name = f'{group_name}_상세'[:31]  # Excel 시트명 길이 제한
                loadings_df.to_excel(writer, sheet_name=sheet_name)
            
            # 3. 그룹간 가중치 비교
            group_summary = []
            for group_name in self.weights.keys():
                total_weight = sum(self.weights[group_name].values())
                avg_weight = np.mean(list(self.weights[group_name].values()))
                max_weight = max(self.weights[group_name].values())
                min_weight = min(self.weights[group_name].values())
                
                group_summary.append({
                    '그룹명': group_name,
                    '변수_수': len(self.weights[group_name]),
                    '총_가중치': total_weight,
                    '평균_가중치': avg_weight,
                    '최대_가중치': max_weight,
                    '최소_가중치': min_weight,
                    '설명분산비율': self.factor_results[group_name]['explained_variance']
                })
            
            group_summary_df = pd.DataFrame(group_summary)
            group_summary_df.to_excel(writer, sheet_name='그룹별_요약', index=False)
        
        print(f"✓ 결과가 {output_file}에 저장되었습니다.")
    
    def visualize_results(self):
        """결과 시각화"""
        n_groups = len(self.factor_results)
        fig, axes = plt.subplots(2, n_groups, figsize=(5*n_groups, 10))
        
        if n_groups == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (group_name, result) in enumerate(self.factor_results.items()):
            # 1. 고유값 그래프
            eigenvalues = result['eigenvalues']
            axes[0, i].plot(range(1, len(eigenvalues)+1), eigenvalues, 'bo-')
            axes[0, i].axhline(y=1, color='r', linestyle='--', alpha=0.7)
            axes[0, i].set_title(f'{group_name}\n고유값 (Eigenvalues)')
            axes[0, i].set_xlabel('요인')
            axes[0, i].set_ylabel('고유값')
            axes[0, i].grid(True, alpha=0.3)
            
            # 2. 가중치 상위 변수들
            weights = self.weights[group_name]
            top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
            
            variables = [item[0] for item in top_weights]
            weight_values = [item[1] for item in top_weights]
            
            axes[1, i].barh(range(len(variables)), weight_values)
            axes[1, i].set_yticks(range(len(variables)))
            axes[1, i].set_yticklabels([v[:20] + '...' if len(v) > 20 else v for v in variables])
            axes[1, i].set_title(f'{group_name}\n상위 가중치 변수')
            axes[1, i].set_xlabel('가중치')
            
            # y축 레이블 뒤집기 (상위부터 표시)
            axes[1, i].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('factor_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ 시각화 결과가 'factor_analysis_results.png'에 저장되었습니다.")

# 사용 예시
def run_weight_analysis(csv_file_path):
    """가중치 분석 실행"""
    
    # 1. 분석기 생성
    calculator = WeightCalculator()
    
    # 2. 데이터 로드 및 전처리
    df = calculator.load_and_preprocess_data(csv_file_path)
    
    # 3. 요인분석 수행
    factor_results = calculator.perform_factor_analysis(df)
    
    # 4. 가중치 계산
    weights = calculator.calculate_weights()
    
    # 5. 결과 저장
    calculator.save_results('신용평가_가중치_결과.xlsx')
    
    # 6. 시각화
    calculator.visualize_results()
    
    # 7. 최종 요약 출력
    print("\n" + "="*60)
    print("최종 가중치 요약")
    print("="*60)
    
    for group_name, group_weights in weights.items():
        print(f"\n[{group_name}] - 총 {len(group_weights)}개 변수")
        top_5 = sorted(group_weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for i, (var, weight) in enumerate(top_5, 1):
            print(f"  {i}. {var}: {weight:.4f}")
    
    return calculator, weights

# 실행 예시
if __name__ == "__main__":
    # CSV 파일 경로를 지정하고 실행
    csv_file_path = "/Users/ahnsaeyeon/guide-on/ai/store_unified_summary.csv"  # 여기에 실제 CSV 파일 경로를 입력하세요
    
    print("신용평가 가중치 분석 시작...")
    print("=" * 60)
    
    try:
        calculator, weights = run_weight_analysis(csv_file_path)
        print("\n✓ 분석이 완료되었습니다!")
        print("✓ 결과 파일: '신용평가_가중치_결과.xlsx'")
        print("✓ 시각화 파일: 'factor_analysis_results.png'")
        
    except FileNotFoundError:
        print(f"❌ 오류: '{csv_file_path}' 파일을 찾을 수 없습니다.")
        print("CSV 파일 경로를 확인해주세요.")
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {str(e)}")
        print("데이터 형식이나 컬럼명을 확인해주세요.")