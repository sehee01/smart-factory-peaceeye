using System;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

public class UIPageCanvasController : MonoBehaviour
{
    public ButtonHandler buttonHandler;

    public GameObject mainCanvas;
    public GameObject listCanvas;
    public GameObject dashboardCanvas;
    public GameObject addListCanvas;
    public GameObject workerInfoCanvas;

    public Button xButton;
    public Button xButton2;
    public Button xButton3;

    [HideInInspector]
    public bool openedUI = false;           // UI창이 열려 있으면 true, 닫혀있으면 false. 마우스로 카메라를 이동시키는 기능을 끌 때 사용.

    /*public Animator mainAnimator;
    public Animator listAnimator;
    public Animator dashboardAnimator;

    private const float transitionTime = 0.5f; // 애니메이션 시간 (초)*/

    void Start()
    {
        openedUI = false;

        mainCanvas.SetActive(true);
        listCanvas.SetActive(false);
        dashboardCanvas.SetActive(false);
        buttonHandler.editPanel?.SetActive(false);
        buttonHandler.addList?.SetActive(false);
        buttonHandler.searchInput?.SetActive(false);
        buttonHandler.alertlog?.SetActive(false);

        xButton.onClick.AddListener(() => XButtonClick());
        xButton2.onClick.AddListener(() => XButton2Click());
    }

    public void ShowWorkerlist()
    {
        openedUI = true;

        listCanvas.SetActive(true);
        mainCanvas.SetActive(false);
        //StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", listCanvas, listAnimator, "SlideIn"));
    }

    public void Showdashboard()
    {
		openedUI = true;

        dashboardCanvas.SetActive(true);
        mainCanvas.SetActive(false);
        //StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", dashboardCanvas, dashboardAnimator, "SlideIn"));
    }
    public void ShowAddlist()
    {
		openedUI = true;

		addListCanvas.SetActive(true);
        listCanvas.SetActive(false);
        //StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", listCanvas, listAnimator, "SlideIn"));
    }

    public void ShowWorkerInfo()
    {
        openedUI = true;

        mainCanvas.SetActive(true);
        workerInfoCanvas.SetActive(true);
    }

    public void XButtonClick()      //workerList -> main
    {
		openedUI = false;

		mainCanvas.SetActive(true);
        if (listCanvas != null) { listCanvas.gameObject.SetActive(false); }
    }
    public void XButton2Click()     //dashboard -> main
    {
		openedUI = false;

		mainCanvas.SetActive(true);
        if (dashboardCanvas != null) { dashboardCanvas.gameObject.SetActive(false); }
    }
    public void XButton3Click()     //addList -> workerList
    {
		listCanvas.SetActive(true);
        if (addListCanvas != null) { addListCanvas.gameObject.SetActive(false); }
    }

    public void XButton4Click()     // workerInfo -> main
    {
		openedUI = false;

		mainCanvas.SetActive(true);
        if (workerInfoCanvas != null) { workerInfoCanvas.SetActive(false); }
    }

    /*private System.Collections.IEnumerator SwitchCanvasWithAnimation(
        GameObject fromCanvas, Animator fromAnimator, string outTrigger,
        GameObject toCanvas, Animator toAnimator, string inTrigger)
    {
        // Start out animation
        fromAnimator.SetTrigger(outTrigger);

        // Wait until out animation finishes
        yield return new WaitForSeconds(transitionTime);

        fromCanvas.SetActive(false);
        toCanvas.SetActive(true);

        // Start in animation
        toAnimator.SetTrigger(inTrigger);
    }*/
}
