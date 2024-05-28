import pandas as pd

data = {
    '학번' : ['1000', '1001', '1002', '1003', '1004', '1005'],
    '이름' : [ 'Steve', 'James', 'Doyeon', 'Jane', 'Pilwoong', 'Tony'],
    '점수': [90.72, 78.09, 98.43, 64.19, 81.30, 99.14]
    }

df = pd.DataFrame(data)
# 앞 부분을 3개만 보기
print(df.head(3))

# 뒷 부분을 3개만 보기
print(df.tail(3))

# '학번'에 해당되는 열을 보기
print(df['학번'])