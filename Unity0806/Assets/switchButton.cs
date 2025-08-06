using UnityEngine;

public class CameraSwitcher : MonoBehaviour
{
    public GameObject OverviewCamera;
    public GameObject Camera;

    private bool isOverview = true;

    public void ToggleCamera()
    {
        isOverview = !isOverview;

        OverviewCamera.SetActive(isOverview);
        Camera.SetActive(!isOverview);
    }
}
