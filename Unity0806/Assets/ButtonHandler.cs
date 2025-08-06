using System.Data;
using Unity.VisualScripting;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class ButtonHandler : MonoBehaviour
{
    //UIPageCanvasControllers에서 가져오기(Show~() 함수들)
    public UIPageCanvasController pageController;

    //캔버스
    public GameObject mainCanvas;
    public GameObject listCanvas;
    public GameObject dashboardCanvas;
    public GameObject buttonCanvas;

    // 패널 및 버튼 목록
    public GameObject Panel;
    public GameObject addList;
    public GameObject searchInput;
    public GameObject alertlog;

    public Button alertButton;
    public Button listButton;
    public Button gotoFirst;
    public Button dashboard;
    public Button exit;

    public Button add;
    public Button edit;
    public Button delete;

    public TMP_InputField searchField;
    public List<GameObject> workerItems;

    // 팝업 관련
    public GameObject popupWindow;
    public TMP_Text popupText;
    public Button popupConfirmButton;

    void Start()
    {
        // 버튼 이벤트 연결 (예시)
        add.onClick.AddListener(() => OnAddClicked());
        edit.onClick.AddListener(() => OnEditClicked());
        delete.onClick.AddListener(() => OnDeleteClicked());

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
        buttonCanvas.SetActive(false);
        Panel?.SetActive(false);
        addList?.SetActive(false);
        searchInput?.SetActive(false);
        alertlog?.SetActive(false);
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
        ShowPopup("종료하시겠습니까?", () =>
        {
            gameObject.SetActive(false);
            // 로그인 화면으로 전환 (가정)
            // SceneManager.LoadScene("LoginScene"); 와 같은 방식 사용 가능
        });
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
        Panel?.SetActive(true);
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

    //대시보드
    public void csvsave()
    {
        Debug.Log("CSV 저장");
        ShowPopup("CSV 저장 경로를 설정하세요", () =>
        {
            string path = "C:\\Users\\sooin\\Desktop\\peaceeyeinstance\\dashboard.csv"; // 테스트용 경로 (실제로는 파일 브라우저 등으로 받는 게 이상적)

            List<string> lines = new List<string>
            {
                "이름,소속팀,위치,근무시간,출근현황,위험도"
                // 예시 데이터 추가 가능
            };

            File.WriteAllLines(path, lines);
            ShowPopup("저장되었습니다.");
        });
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
