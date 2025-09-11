from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pymysql

def get_db_connection():
    """DB 연결"""
    return pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='Ab1515!!',
        db='guideon',
        charset='utf8mb4',
        use_unicode=True,
    )

# 1️⃣ 모델과 토크나이저 불러오기 (한국어 감정분석)
model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-v2022")
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")

# model = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base-finetuned-nsmc")
# tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-finetuned-nsmc")

# 2️⃣ DB에서 리뷰 데이터 가져오기
def get_reviews_from_db():
    """MySQL DB에서 googlemaps_reviews의 content 컬럼을 읽어옴"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT content FROM googlemaps_reviews WHERE content IS NOT NULL AND content != 'N/A'"
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        # 튜플에서 문자열만 추출
        reviews = [row[0] for row in results if row[0] and row[0].strip()]
        return reviews
        
    except Exception as e:
        print(f"DB 연결 오류: {e}")
        # DB 연결 실패시 예시 데이터 사용
        return [
            "명동교자의 칼국수는 국물이 진하고 깊은 맛이 난다. 정말 맛있다!",
            "서비스가 별로였고, 김치 맛도 마음에 들지 않았다.",
            "만두가 아주 맛있어서 만족스럽다."
        ]

# 3️⃣ DB에서 rating 평균 구하기
def get_rating_average():
    """MySQL DB에서 googlemaps_reviews의 rating 컬럼 평균을 구함"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = "SELECT AVG(rating) FROM googlemaps_reviews WHERE rating IS NOT NULL"
        cursor.execute(query)
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if result and result[0] is not None:
            return round(float(result[0]), 2)
        else:
            return None
            
    except Exception as e:
        print(f"DB 연결 오류: {e}")
        return None

# DB에서 리뷰 데이터 가져오기
reviews = get_reviews_from_db()
print(f"DB에서 {len(reviews)}개의 리뷰를 가져왔습니다.")

# DB에서 rating 평균 구하기
rating_avg = get_rating_average()
if rating_avg is not None:
    print(f"DB의 평균 별점: {rating_avg}점")
else:
    print("평균 별점을 구할 수 없습니다.")



# 3️⃣ 토큰화 및 모델 입력
inputs = tokenizer(reviews, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

# 모델 출력 구조 확인
# print("Model config:", model.config)
# print("Logits shape:", outputs.logits.shape)
# print("Raw logits (첫 3개):")
# print(outputs.logits[:3])
# print("Label mapping:", model.config.id2label if hasattr(model.config, 'id2label') else "없음")

# 토큰화 결과 확인
# sample_text = reviews[0]
# tokens = tokenizer.tokenize(sample_text)
# print(f"원문: {sample_text}")
# print(f"토큰: {tokens}")
# print(f"토큰 개수: {len(tokens)}")

# 4️⃣ 긍정/부정 확률 계산
probs = torch.softmax(outputs.logits, dim=1)  # [:,0]=부정, [:,1]=긍정

# 5️⃣ 개별 리뷰 판정
predictions = ["긍정" if p[1] > 0.5 else "부정" for p in probs]

# 6️⃣ 전체 긍정 비중 계산
positive_ratio = (probs[:,1] > 0.5).sum().item() / len(reviews)


# 진단 코드
# print("=== 모델 정보 ===")
# print(f"모델 클래스: {type(model)}")
# print(f"설정: {model.config}")

# print("\n=== 첫 번째 리뷰 상세 분석 ===")
# sample_input = tokenizer(reviews[0], return_tensors="pt")
# sample_output = model(**sample_input)
# print(f"입력 토큰 ID: {sample_input['input_ids']}")
# print(f"Raw logits: {sample_output.logits}")
# print(f"Softmax 결과: {torch.softmax(sample_output.logits, dim=1)}")

# 7️⃣ 결과 출력
for review, pred, prob in zip(reviews, predictions, probs):
    print(f"리뷰: {review}\n예측: {pred}, 긍정 확률: {prob[1]:.2f}\n")

print(f"전체 리뷰 중 긍정 비중: {positive_ratio:.2f}")

# API에서 사용할 수 있도록 positive_ratio와 rating_avg 값 출력
if __name__ == "__main__":
    import json
    result = {
        "positive_ratio": positive_ratio,
        "rating_average": rating_avg
    }
    print(json.dumps(result))
    