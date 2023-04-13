## 0412 update 내용

- code 폴더안에 있는 genre_transition 모듈을 사용하면 음원데이터들을 분석용 데이터로 변환시킬수 있습니다.
  - 이 때, gtzan 데이터와 같이 대상폴더 > 장르특성 > 음원파일 순으로 정리해주세요
  - 간혹 .DS_store파일이 생성되어서 에러가나는 경우가 있습니다. github과 연동된 폴더에서 사용하는 경우 .DS_store파일을 삭제하고, 생성되지 않게 설정해주세요.
  - 참고 : http://uidesignguides.com/mac-ds-store-file-prevent/
  - feature_extraction_by_folder 결과의 첫 번째와 mfcc_extraction_by_folder의 첫 번째를 각각 데이터 프레임으로 변환한뒤, 합쳐주면 됩니다.
  
 - downloading_youtube 파일에는 유튜브링크와 재생목록으로 음원을 다운로드하는 코드가 있습니다.
  - 다운로드된 음원데이터들은 아마 mp4로 저장되었을겁니다.
  - wav파일로 변환해서 사용해주세요. mp3로 변환하는 경우, librosa에서 종종 에러가 발생해 값에 오류가 나타납니다.
 
