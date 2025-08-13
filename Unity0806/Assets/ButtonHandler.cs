using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Runtime.InteropServices;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;
using System.Linq;

public class ButtonHandler : MonoBehaviour
{
    //UIPageCanvasControllers에서 가져오기(Show~() 함수들)
    public UIPageCanvasController pageController;

    [Header("캔버스")]
    public GameObject mainCanvas;
    public GameObject listCanvas;
    public GameObject dashboardCanvas;

    [Header("패널 목록")]
    //메인화면
    public GameObject alertlog;
    public GameObject LogoutPanel;

    //작업자 리스트
    public GameObject editPanel;
    public GameObject addList;
    public GameObject searchInput;
    public GameObject filterPanel;

    public TMP_InputField searchField;
    public List<GameObject> workerItems;

    [Header("버튼 목록")]
    //메인화면
    public Button alertButton;
    public Button listButton;
    public Button gotoFirst;
    public Button dashboard;
    public Button exit;
    public Button accept;  //로그아웃 확인
    public Button cancel;  //로그아웃 취소

    //작업자 리스트
    public Button add;
    public Button edit;
    public Button delete;
    public Button filter;
    public Button upperlower;  //오름차순, 다시 누르면 내림차순
    
    //대시보드
    public GameObject saveFile;
    public Button acceptSave;
    public Button cancelSave;

    // 팝업 관련
    public GameObject popupWindow;
    public TMP_Text popupText;
    public Button popupConfirmButton;

    // --- 정렬 기능에 필요한 변수들 (새로 추가) ---
    private string currentSortColumn;   // 현재 정렬 기준이 되는 열 이름
    private bool isSortAscending = true; // 현재 정렬 방향 (true: 오름차순, false: 내림차순)

    // --- 자바스크립트 플러그인 함수 임포트 --- WebGL 빌드에서만 이 함수를 인식합니다.
    [DllImport("__Internal")]
    private static extern void DownloadFile(string fileName, string fileContent);

    void Start()
    {
        // 버튼 이벤트 연결 (예시)
        add.onClick.AddListener(() => OnAddClicked());
        edit.onClick.AddListener(() => OnEditClicked());
        delete.onClick.AddListener(() => OnDeleteClicked());
        acceptSave.onClick.AddListener(() => OnAcceptSaveClicked());
        cancelSave.onClick.AddListener(() => OnCancelSaveClicked());

        //alertButton.onClick.AddListener(() => OnAlertButtonClicked());
        listButton.onClick.AddListener(() => OnListButtonClicked());
        //gotoFirst.onClick.AddListener(() => OnGotoFirstClicked());
        //dashboard.onClick.AddListener(() => OnDashboardClicked());
        //exit.onClick.AddListener(() => OnExitClicked());

        //popupConfirmButton.onClick.AddListener(() => popupWindow.SetActive(false));
        
        // --- 정렬 버튼 이벤트 리스너 연결 (새로 추가) ---
        upperlower.onClick.AddListener(OnUpperLowerClicked);

        // --- 정렬 기준 초기화 (새로 추가) ---
        if (columnNames.Count > 0)
        {
            currentSortColumn = columnNames[0]; // 기본 정렬 기준을 '이름'으로 설정
        }
    }

    //메인 화면
    public void OnGotoFirstClicked()
    {
        Debug.Log("go to main");
        mainCanvas.SetActive(true);
        listCanvas.SetActive(false);
        dashboardCanvas.SetActive(false);
        saveFile.SetActive(false);
        editPanel?.SetActive(false);
        addList?.SetActive(false);
        searchInput?.SetActive(false);
        //alertlog?.SetActive(false);
        LogoutPanel.SetActive(false);
    }

    public void OnListButtonClicked()
    {
        Debug.Log("WorkerList click");
        //캔버스 이동: mainCanvas -> listCanvas
        pageController.ShowWorkerlist();
        /*mainCanvas.SetActive(false);
        listCanvas.SetActive(true);*/
    }

    public void OnDashboardClicked()
    {
        Debug.Log("Dashboard click");
        //캔버스 이동: mainCanvas -> dashboardCanvas
        pageController.Showdashboard();
        /*mainCanvas.SetActive(false);
        dashboardCanvas.SetActive(true);*/
    }

    public void OnExitClicked()
    {
        Debug.Log("Exit click");
        LogoutPanel.SetActive(true);
    }

    public void CancelClick()
    {
        // 로그아웃 취소: 패널 닫기
        LogoutPanel.SetActive(false);
        Debug.Log("로그아웃 취소함");
    }

    public void AcceptClick()
    {
        // 로그아웃 확인: 로그아웃 로직 구현
        OnLogoutConfirmed();
        LogoutPanel.SetActive(false);
        Debug.Log("로그아웃 승인함");
    }

    public void OnLogoutConfirmed()
    {
#if UNITY_WEBGL == true && !UNITY_EDITOR
        // WebGL 빌드 환경에서만 JavaScript 함수를 호출합니다.
        //Application.ExternalCall("handleUnityLogout");
#else
        // 에디터 환경에서는 로그만 출력하여 테스트합니다.
        Debug.Log("에디터 모드: 로그아웃 신호를 보냈습니다.");
#endif
    }

    public void OnAlertButtonClicked()
    {
        Debug.Log("알림 버튼 클릭");
        // alertlog가 현재 활성화되어 있는지(.activeSelf) 확인합니다.
        bool isCurrentlyActive = alertlog.activeSelf;
        // 켜져 있으면 -> !true -> false (꺼짐)
        // 꺼져 있으면 -> !false -> true (켜짐)
        alertlog.SetActive(!isCurrentlyActive);
    }

    //작업자 리스트
    public void editing()
    {
        Debug.Log("editing click");
        editPanel?.SetActive(true);
    }

    private void OnAddClicked()
    {
        Debug.Log("Add click");

        pageController.ShowAddlist();
        //addList?.SetActive(true);

        listCanvas?.SetActive(false);
    }

    private void OnEditClicked()
    {
        Debug.Log("기존 항목 수정");

        pageController.ShowAddlist();
        //addList?.SetActive(true);

        listCanvas?.SetActive(false);
    }

    private void OnDeleteClicked()
    {
        Debug.Log("삭제하시겠습니까?");
        ShowPopup("삭제하시겠습니까?", () =>
        {
            // 삭제 로직
            foreach (var item in workerItems)
            {
                Debug.Log("DeleteClick");
                //Destroy(item);
            }
            ShowPopup("삭제되었습니다.");
            listCanvas?.SetActive(true);
        });
    }

    public void search( )
    {
        Debug.Log("search click");
        // 1. 검색창(searchInput)의 현재 활성화 상태를 확인합니다.
        bool isSearchActive = searchInput.activeSelf;
        // 2. 현재 상태의 반대 값으로 설정하여 토글을 구현합니다.
        searchInput.SetActive(!isSearchActive);
        // 3. 만약 검색창을 끈 것이라면(isSearchActive가 true였다면),
        //    검색어를 초기화하고 리스트를 원래대로 되돌립니다.
        if (isSearchActive)
        {
            Debug.Log("검색창을 닫고 리스트를 초기화합니다.");
            searchField.text = ""; // 검색 필드 초기화
            // 모든 아이템을 다시 보여주기
            foreach (var item in workerItems)
            {
                item.SetActive(true);
            }
        }
    }

    // 이 함수는 Input Field의 OnValueChanged 이벤트에 연결하면 실시간 검색이 가능해집니다.
    public void UpdateSearchByKeyword()
    {
        string keyword = searchField.text.ToLower();
        // 대소문자 구분 없이 검색하려면 ToLower()를 사용합니다.
       foreach (var item in workerItems)
        {
            // workerItems의 각 항목에서 '이름' 자식 오브젝트의 TMP_Text 값을 가져와 비교
            string itemName = GetColumnValue(item, "이름").ToLower();
            bool contains = itemName.Contains(keyword);
            item.SetActive(contains);
        }
    }
    
    // 정렬 가능한 열 이름들 (UI Text 오브젝트 이름과 동일)
    public List<string> columnNames = new List<string>
    {
        "이름", "소속", "현재위치", "근무시간", "출근현황", "위험도"
    };

    public void Filter()
    {
        //각 열에 들어가는 내용(이름, 소속팀, 현재위치, 근무시간, 출근현황, 위험도 등) 중
        //원하는 자료만을 (모두) 선택하여
        //선택된 자료만 리스트에 출력
        // 실제 구현은 사용자 선택 UI (토글, 체크박스 등)를 통해 필터 조건 지정
        Debug.Log("필터 기능 실행 - 조건에 맞는 항목만 출력");
        filterPanel.SetActive(!filterPanel.activeSelf);

    }

    // --- 실제 정렬을 실행하는 함수 (새로 추가) ---
    private void SortByColumn(string columnName, bool ascending)
    {
        // 1. workerItems 리스트를 정렬합니다.
        // 정렬 기준(columnName)에 해당하는 자식 오브젝트의 TMP_Text 값을 가져와서 비교합니다.
        workerItems = ascending
            ? workerItems.OrderBy(w => GetColumnValue(w, columnName)).ToList()
            : workerItems.OrderByDescending(w => GetColumnValue(w, columnName)).ToList();

        // 2. 정렬된 순서대로 UI 계층 구조를 재배치합니다.
        // SetSiblingIndex는 부모 Transform 내에서 순서를 변경하여 화면에 보이는 순서를 바꿉니다.
        for (int i = 0; i < workerItems.Count; i++)
        {
            workerItems[i].transform.SetSiblingIndex(i);
        }
    }

    // --- 정렬 기준 값을 가져오는 헬퍼 함수 (새로 추가) ---
    private string GetColumnValue(GameObject workerItem, string columnName)
    {
        // workerItem(한 행)의 자식 중 columnName과 이름이 같은 오브젝트를 찾습니다.
        Transform columnTransform = workerItem.transform.Find(columnName);
        if (columnTransform != null)
        {
            // 찾은 오브젝트에서 TMP_Text 컴포넌트의 텍스트를 반환합니다.
            TMP_Text textComp = columnTransform.GetComponent<TMP_Text>();
            if (textComp != null)
            {
                return textComp.text;
            }
        }
        return ""; // 값을 찾지 못하면 빈 문자열을 반환합니다.
    }

    // --- 오름차순/내림차순 버튼 클릭 시 호출될 함수 (새로 추가) ---
    private void OnUpperLowerClicked()
    {
        // 실제 정렬 함수를 호출합니다.
        SortByColumn(currentSortColumn, isSortAscending);

        Debug.Log($"{currentSortColumn} 기준으로 {(isSortAscending ? "오름차순" : "내림차순")} 정렬 완료");

        // 다음 클릭을 위해 정렬 방향을 반전시킵니다.
        isSortAscending = !isSortAscending;
    }

    //대시보드 --- csvsave() 함수 수정 ---
    public void csvsave()
    {
        Debug.Log("CSV 저장 시도");
        saveFile?.SetActive(true);
        if (saveFile)
        {
            OnAcceptSaveClicked();
        } else
        {
            OnCancelSaveClicked();
        }
    }
        
    public void OnAcceptSaveClicked() {
        Debug.Log("svcSaved");
        // 1. CSV로 만들 데이터 생성
        // 헤더(첫 줄)와 데이터 예시를 만듭니다. 실제 데이터로 교체해야 합니다.
        List<string> lines = new List<string>
        {
            "이름,소속팀,위치,근무시간,출근현황,위험도", // CSV 헤더
            "홍길동,A팀,작업장1,8시간,출근,낮음",      // 예시 데이터 1
            "김철수,B팀,작업장2,5시간,출근,보통"       // 예시 데이터 2
        };
        string csvContent = string.Join("\n", lines); // 각 줄을 줄바꿈 문자로 연결

        // 2. 플랫폼에 따라 다른 저장 방식 실행
#if UNITY_WEBGL && !UNITY_EDITOR
        // WebGL 빌드 환경일 경우:
        // 위에서 만든 자바스크립트 플러그인의 DownloadFile 함수를 호출합니다.
        // 첫 번째 인자는 파일 이름, 두 번째 인자는 파일 내용입니다.
        DownloadFile("dashboard.csv", csvContent);
        ShowPopup("브라우저에서 다운로드가 시작됩니다.");
#else
        // Unity 에디터 또는 PC 빌드 환경일 경우:
        // 기존처럼 직접 파일 경로에 저장합니다. (테스트용)
        // 실제 배포 시에는 파일 탐색기를 여는 코드로 바꾸는 것이 좋습니다.
        string path = "C:\\Users\\sooin\\Desktop\\peaceeyeinstance\\dashboard.csv";
        try
        {
            File.WriteAllText(path, csvContent);
            ShowPopup($"저장되었습니다.\n경로: {path}");
        }
        catch (System.Exception e)
        {
            ShowPopup($"저장에 실패했습니다: {e.Message}");
            Debug.LogError($"파일 저장 오류: {e.Message}");
        }
#endif
    }

    public void OnCancelSaveClicked()
    {
        saveFile?.SetActive(false);
    }

    // 팝업 유틸리티
    private void ShowPopup(string message, System.Action onConfirm = null)
    {
        Debug.Log("showPopup");
        popupWindow?.SetActive(true); 
        popupText.text = message;

        popupConfirmButton.onClick.RemoveAllListeners();
        popupConfirmButton.onClick.AddListener(() =>
        {
            popupWindow.SetActive(false);
            onConfirm?.Invoke();
        });
    }
}
