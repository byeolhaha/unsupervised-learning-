# 비지도 학습을 통한 정제 알약 형상 분류 (Classification of tablets and pills by unsupervised learning)

## Description
> 2018.3.~2020.7. (졸업 논문)

### Purpose
- 비지도 학습을 이용하여 라벨이 없는 대량의 데이터 
즉 국내 정제 얄약의 형상에 대한 구조 및 분류 기준을 알아보는 것을 목적으로 한다. 
 또한 향후 진행될 지도 학습에 명확한 표본 제공
### Experimental
- 미국의 IOM(Institute of Medicine)에 따르면 미국에서 한 해 약 10만 명이 약화사고로 인해 죽고 그 중 의약품 사용 과오로 인한 사고는 45%를 차지한다고 한다.
이는 약품 과오로써 조제 사고를 의미하며, 약의 이름 , 포장, 모양의 혼동에 의해 발생한다. 
그러나 현재 약의 모양 즉 어떤 형태적 요인이 혼동을 불러오는지에 대한 연구가 부족하다. 
국내 정제 알약은 대략 16,128개가 존재하며 또한 매년 신약이 추가되고 있다. 
이는 대량의 데이터이기 때문에 형상에 대한 라벨 데이터가 존재하지 않고 존재하기도 어렵다.
따라서 비지도 학습을 통해 데이터의 구조를 알아보고 어떤 기준을 중심으로 약물의 형상을 분류하는지 알아보고자 한다.
### Materials and Mehtods
- 본 연구에서는 알약 형상 데이터들을 바탕으로 비지도 학습을 돌려 유사한 알약으로 인지하는 기준과 유사한 알약의 그룹을 확인하였다.
의약품 안전나라에서 추출된 의약품 이미지 16,128장을 알약 형상 데이터의 대상으로 선정하였다. 
분석 방법은 크게 두 가지로 진행되었다.
- 첫 번째 방법은 PCA를 진행하여 특성을 추출하고 이에 K-means를 진행하여 유사한 알약의 그룹을 확인하였다. **(1차 비지도 학습)**
- 두 번째 방법은 첫 번째 방법의 단점을 보완하는 방법을 택하여 t-SNE와 DBSCAN을 이용하였다. **(2차 비지도 학습)**
 대량의 이미지 데이터의 차원의 수가 고차원이기 때문에 데이터를 잘 설명할 수 있는 저차원으로 바꾸는 t-SNE를 이용하였다.
 이에 DBSCAN을 진행하여 클러스터 밖에 존재하는 특이한 데이터를 제외시켰다. 

### Flow Chart
![image](https://user-images.githubusercontent.com/108210958/224484791-704cf9b9-fbd8-47b7-89cb-69938f722da9.png)

### Data Normalization
![image](https://user-images.githubusercontent.com/108210958/224485112-15eb9b89-4e62-464d-8def-fad6a351e8aa.png)

## Results
- 비지도 학습의 결과 크게 색, 크기, 모양에 대한 분류 기준이 등장하였다.
- 구체적으로 색에 대해서는 흰색, 붉은 색, 노란 색, 파란 색, 경질 캡슐(알약 가운데를 기준으로 색이 다른), 
- 크기에 대해서는 큰 원형, 작은 원형, 큰 타원형, 작은 타원형, 모양에 대해서는 원형, 타원형, 연질 캡슐이 존재하였다. 
- 또한 PCA와 k-means 보다는 IPCA, t-SNE, DBSCAN의 방법이 국내 정제 알약에 대한 비지도 학습 방법으로 더 적합하였다. 

### 1차 비지도 학습
![image](https://user-images.githubusercontent.com/108210958/224485534-21d5a8ba-8932-480c-bc65-3c8e6239109c.png)

![image](https://user-images.githubusercontent.com/108210958/224485593-077f43f0-11c6-4d66-bd0e-273260578e85.png)

![image](https://user-images.githubusercontent.com/108210958/224485606-42769e93-c123-46e2-a9a2-5cddba64a9b0.png)

![image](https://user-images.githubusercontent.com/108210958/224485620-501d0744-cafb-42a9-851c-e690bb1f3b9d.png)


▲ 좌표상에 명확하게 구별되는 특이한 데이터가 존재

그래서 이 특이데이터를 제거하고 다시 한번 1치 비지도 학습을 진행하였다.

![image](https://user-images.githubusercontent.com/108210958/224485669-94e90b16-3b7a-4ecd-b57f-aa573558354d.png)

![image](https://user-images.githubusercontent.com/108210958/224485690-57090a2d-3245-4aa9-8c5b-ed36b294bb58.png)

![image](https://user-images.githubusercontent.com/108210958/224485699-379ee3ac-07eb-4eeb-83df-7d712f2892d3.png)

![image](https://user-images.githubusercontent.com/108210958/224485720-316ef3b1-4bfd-47b1-95ee-b12495cf0f8d.png)

▲ PCA를 통해서 특성을 반영한 엘보우 방법(Elbow method)으로 그룹의 개수는 4개가 적절한 것으로 나타남
![image](https://user-images.githubusercontent.com/108210958/224485754-3f6b4fab-4522-4bcb-961e-79225b716f6c.png)

▲ PCA로 특성을 추출한 데이터를 k-means를 통해서 클러스터의 개수를 4개부터 9개까지의 덴드로그램
![image](https://user-images.githubusercontent.com/108210958/224485774-34c597f0-9165-4871-94eb-79fe229fe16b.png)

▲ PCA로 특성을 추출한 데이터를 k-means를 통해서 클러스터의 개수를 4개부터 9개까지 등장한 분류 기준

### 2차 비지도 학습
![image](https://user-images.githubusercontent.com/108210958/224486666-cf381c82-52e1-4d17-80d4-c57821512d30.png)

▲ IPCA를 통해서 160차원의 특성을 추출하고 t-SNE를 통해 2차원으로 시각화 한 결과

![image](https://user-images.githubusercontent.com/108210958/224486757-2f755eaa-7644-4552-8f5b-3d4828c266fe.png)

▲ DBSCAN에 의해 얻어진 클러스터 113개를 5가지 기준에 따라서 5개의 클러스터로 나눈 결과

------------------------------------------------------
#### 유색이면서 동그란 알약의 PCA 3차원 결과
![image](https://user-images.githubusercontent.com/108210958/224487149-e3727022-c0d0-4e72-a028-ce57ad750f71.png)
![image](https://user-images.githubusercontent.com/108210958/224487153-7e68c942-b9ed-4a63-8c91-1869a4926888.png)
![image](https://user-images.githubusercontent.com/108210958/224487156-dd916f5c-2d8b-495b-8099-87911da65998.png)
-----------------------------------------------------
#### 흰색이면서 동그란 알약의 PCA 3차원 결과
![image](https://user-images.githubusercontent.com/108210958/224487174-f423a6c6-1b07-488a-8029-82648cca816d.png)
![image](https://user-images.githubusercontent.com/108210958/224487179-0d1253c8-ff56-4fb1-8e1f-bef514dec2a9.png)
![image](https://user-images.githubusercontent.com/108210958/224487181-23d80ed5-a8a7-49f6-aa6d-45ef6b8e0fad.png)
---------------------------------------------------------
#### 흰색이면서 동그란 알약의 글자 인식 PCA 3차원 결과
![image](https://user-images.githubusercontent.com/108210958/224487201-e1e3e58b-1948-4646-aa84-0b9ab0e5ec46.png)
![image](https://user-images.githubusercontent.com/108210958/224487206-fc7e2559-e7ae-45b3-b44c-b1709400a85e.png)
![image](https://user-images.githubusercontent.com/108210958/224487210-4d5b173c-64d4-4e14-b8e2-64f7144bb4c2.png)
-----------------------------------------------------------
#### 경질 캡슐의 PCA 3차원 결과
![image](https://user-images.githubusercontent.com/108210958/224487224-57fa4734-01c0-44aa-885d-dde8ceb75e66.png)
![image](https://user-images.githubusercontent.com/108210958/224487228-51e6eb2e-bd21-4ee6-9677-599672db0ce7.png)
![image](https://user-images.githubusercontent.com/108210958/224487233-dfa85cc7-b7d8-4abd-beb0-42c6e3bc328f.png)
------------------------------------------------------
#### 흰색이면서 납작한 알약의 PCA 3차원 결과
![image](https://user-images.githubusercontent.com/108210958/224487247-ddc0996c-79f2-4084-b70c-85d520c6f8df.png)
![image](https://user-images.githubusercontent.com/108210958/224487253-b8b66053-00ae-4db3-a99e-e9d1443f4f11.png)
![image](https://user-images.githubusercontent.com/108210958/224487254-e7e020f8-4d65-4f59-885f-2df2216b1f32.png)
------------------------------------------------------------
#### 유색이면서 납작한 알약의 PCA 3차원 결과
![image](https://user-images.githubusercontent.com/108210958/224487274-6f807586-91f9-4828-8d82-4e9fa236b9e0.png)
![image](https://user-images.githubusercontent.com/108210958/224487279-5965c433-e064-4421-bd8f-323c072e2133.png)
![image](https://user-images.githubusercontent.com/108210958/224487284-b8528a87-72c0-4433-b45e-36aabba8b9cf.png)





