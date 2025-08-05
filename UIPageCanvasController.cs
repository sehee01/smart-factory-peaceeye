using UnityEngine;

public class UIPageCanvasController : MonoBehaviour
{
    public GameObject mainCanvas;
    public GameObject listCanvas;
    public GameObject dashboardCanvas;

    public Animator mainAnimator;
    public Animator listAnimator;
    public Animator dashboardAnimator;

    private const float transitionTime = 0.5f; // 애니메이션 시간 (초)

    void Start()
    {
        //mainCanvas.SetActive(true);
        //listCanvas.SetActive(false);
        //dashboardCanvas.SetActive(false);
    }

    public void Showlist()
    {
        StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", listCanvas, listAnimator, "SlideIn"));
    }

    public void Showdashboard()
    {
        StartCoroutine(SwitchCanvasWithAnimation(mainCanvas, mainAnimator, "SlideOut", dashboardCanvas, dashboardAnimator, "SlideIn"));
    }

    public void Closelist()
    {
        StartCoroutine(SwitchCanvasWithAnimation(listCanvas, listAnimator, "SlideOut", mainCanvas, mainAnimator, "SlideIn"));
    }

    public void Closedashboard()
    {
        StartCoroutine(SwitchCanvasWithAnimation(dashboardCanvas, dashboardAnimator, "SlideOut", mainCanvas, mainAnimator, "SlideIn"));
    }

    private System.Collections.IEnumerator SwitchCanvasWithAnimation(
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
    }
}
