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

    public Button xButton;
    public Button xButton2;

    /*public Animator mainAnimator;
    public Animator listAnimator;
    public Animator dashboardAnimator;

    private const float transitionTime = 0.5f; // 애니메이션 시간 (초)*/

    void Start()
    {
        mainCanvas.SetActive(true);
        listCanvas.SetActive(false);
        dashboardCanvas.SetActive(false);
        buttonHandler.buttonCanvas.SetActive(false);
        buttonHandler.Panel?.SetActive(false);
        buttonHandler.addList?.SetActive(false);
        buttonHandler.searchInput?.SetActive(false);
        buttonHandler.alertlog?.SetActive(false);

        xButton.onClick.AddListener(() => XButtonClick());
        xButton2.onClick.AddListener(() => XButton2Click());
    }

    public void ShowWorkerlist()
    {
        listCanvas.SetActive(true);
        mainCanvas.SetActive(false);
        //StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", listCanvas, listAnimator, "SlideIn"));
    }

    public void Showdashboard()
    {
        dashboardCanvas.SetActive(true);
        mainCanvas.SetActive(false);
        //StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", dashboardCanvas, dashboardAnimator, "SlideIn"));
    }
    public void ShowAddlist()
    {
        addListCanvas.SetActive(true);
        listCanvas.SetActive(false);
        //StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", listCanvas, listAnimator, "SlideIn"));
    }

    public void XButtonClick()
    {
        mainCanvas.SetActive(true);
        if (listCanvas != null) { listCanvas.gameObject.SetActive(false); }
        if (dashboardCanvas != null) { dashboardCanvas.gameObject.SetActive(false); }
    }
    public void XButton2Click()
    {
        listCanvas.SetActive(true);
        if (addListCanvas != null) { addListCanvas.gameObject.SetActive(false); }
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
