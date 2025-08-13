using UnityEngine;

public class DashboardOpener : MonoBehaviour
{
    public GameObject Client;
    public DashboardClient dashboardClient;

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {

    }

    private void OnEnable()
    {
        StartCoroutine(dashboardClient.FetchOngoingTasks());
        StartCoroutine(dashboardClient.FetchPartSummary());
        StartCoroutine(dashboardClient.FetchWorkerStatus());
        StartCoroutine(dashboardClient.FetchRecentAlerts(72));
        StartCoroutine(dashboardClient.FetchRiskSummary());
        StartCoroutine(dashboardClient.FetchZoneSummary());
    }
}
