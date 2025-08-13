mergeInto(LibraryManager.library, {
  DownloadFile: function(fileNamePtr, fileContentPtr) {
    var fileName = UTF8ToString(fileNamePtr);
    var fileContent = UTF8ToString(fileContentPtr);

    // 1. Blob 객체 생성
    // Blob은 파일과 유사한 객체로, 텍스트나 바이너리 데이터를 담을 수 있습니다.
    var blob = new Blob([fileContent], { type: "text/csv;charset=utf-8;" });

    // 2. 다운로드를 위한 임시 링크(<a> 태그) 생성
    var link = document.createElement("a");

    // 3. 생성된 Blob 객체를 가리키는 URL 생성
    var url = URL.createObjectURL(blob);
    link.href = url;
    link.download = fileName; // 다운로드 시 사용될 파일 이름 설정

    // 4. 링크를 보이지 않게 처리하고 문서에 추가
    link.style.display = "none";
    document.body.appendChild(link);

    // 5. 프로그래밍 방식으로 링크 클릭하여 다운로드 실행
    link.click();

    // 6. 사용이 끝난 임시 객체들 정리
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
});