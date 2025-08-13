using System.Collections.Generic;
using System.Data;
using System.IO;
using System.Runtime.InteropServices;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class ButtonHandler : MonoBehaviour
{
    //UIPageCanvasControllers에서 가져오기(Show~() 함수들)
    public UIPageCanvasController pageController;

    //캔버스
    public GameObject mainCanvas;
    public GameObject listCanvas;
    public GameObject dashboardCanvas;

    // 패널 및 버튼 목록
    public GameObject editPanel;
    public GameObject addList;
    public GameObject searchInput;
    public GameObject alertlog;
    public GameObject LogoutPanel;

    public Button alertButton;
    public Button listButton;
    public Button gotoFirst;
    public Button dashboard;
    public Button exit;
    public Button accept;
    public Button cancel;

    public Button add;
    public Button edit;
    public Button delete;

    public GameObject saveFile;
    public Button acceptSave;
    public Button cancelSave;

    public TMP_InputField searchField;
    public List<GameObject> workerItems;

    // 팝업 관련
    public GameObject popupWindow;
    public TMP_Text popupText;
    public Button popupConfirmButton;

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

        alertButton.onClick.AddListener(() => OnAlertButtonClicked());
        listButton.onClick.AddListener(() => OnListButtonClicked());
        gotoFirst.onClick.AddListener(() => OnGotoFirstClicked());
        dashboard.onClick.AddListener(() => OnDashboardClicked());
        exit.onClick.AddListener(() => OnExitClicked());
        //popupConfirmButton.onClick.AddListener(() => popupWindow.SetActive(false));
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
        alertlog?.SetActive(false);
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
        Application.ExternalCall("handleUnityLogout");
#else
        // 에디터 환경에서는 로그만 출력하여 테스트합니다.
        Debug.Log("에디터 모드: 로그아웃 신호를 보냈습니다.");
#endif
    }

    public int count = 0;

    public void OnAlertButtonClicked()
    {
        count++;
        Debug.Log("알림 내역 클릭함");
        alertlog?.SetActive(true);
    }

    public void alertlogOpen()
    {
        Debug.Log("알람 로그 출력됨");
        if (count == 1 && alertlog == true)
        {
            //알림 내역 한 번 더 클릭하면 로그 꺼짐
            alertlog.SetActive(false);
            Debug.Log("count를 초기화합니다.");
            count = 0;
        }
        //알림 내역 로그 출력됨
        //아직 최신 알람 최상단에 띄우는 건 안됨
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
        searchInput?.SetActive(true);
        string keyword = searchField?.text;

        foreach (var item in workerItems)
        {
            bool contains = item.name.Contains(keyword);
            item.SetActive(contains);
        }
    }

    public void filter()
    {
        //각 열에 들어가는 내용(이름, 소속팀, 현재위치, 근무시간, 출근현황, 위험도 등) 중
        //원하는 자료만을 (모두) 선택하여
        //선택된 자료만 리스트에 출력
        // 실제 구현은 사용자 선택 UI (토글, 체크박스 등)를 통해 필터 조건 지정
        Debug.Log("필터 기능 실행 - 조건에 맞는 항목만 출력");
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
