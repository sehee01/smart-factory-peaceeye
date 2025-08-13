using UnityEngine;
using UnityEngine.Networking;
using System.Collections;
using System.Collections.Generic;
using System;
using TMPro;


#region DTOs (서버 JSON 키와 정확히 일치해야 함)

// 1) /dashboard/ongoing-tasks
[System.Serializable] public class SummaryV1 { public TaskV1[] ongoing_tasks; }
[System.Serializable]
public class TaskV1
{
    public string task_name; public string part; public string due_date;
    public int progress; public bool is_delayed;
}

// 2) /dashboard/part-summary
[System.Serializable] public class PartSummaryRoot { public PartSummaryItem[] part_summary; }
[System.Serializable]
public class PartSummaryItem
{
    public string part; public int task_count; public float delay_rate; // 0~1
}

// 3) /dashboard/worker-status
[System.Serializable] public class WorkerStatusRoot { public WorkerStatusItem[] worker_status; }
[System.Serializable]
public class WorkerStatusItem
{
    public string worker_id; public int total_tasks;
}

// 4) /dashboard/recent-alerts?hours=72
[System.Serializable] public class RecentAlertsRoot { public RecentAlertItem[] recent_alerts; public int hours; }
[System.Serializable]
public class RecentAlertItem
{
    public string worker_id; public string type; public int count;
}

// 5) /dashboard/risk-summary
[System.Serializable] public class RiskSummaryRoot { public RiskSummary risk_summary; }
[System.Serializable]
public class RiskSummary
{
    public int total_workers; public float ppe_violation_rate; public int roi_violation_count;
}

// 6) /dashboard/zone-summary
[System.Serializable] public class ZoneSummaryRoot { public ZoneSummaryItem[] zone_summary; }
[System.Serializable]
public class ZoneSummaryItem
{
    public string zone_id; public int active_workers;
}

#endregion

public class DashboardClient : MonoBehaviour
{
    [Header("HTTP Base URL")]
    [SerializeField] private string baseUrl = "http://localhost:5000";

    [Header("Polling (sec). 0 = one-shot")]
    [SerializeField] private float pollIntervalSec = 0f;

    //private bool running;

    //private void OnEnable() { running = true; StartCoroutine(PollLoop()); }
    //private void OnDisable() { running = false; }
    //private void OnDestroy() { running = false; }

    // 다른 스크립트가 구독할 콜백
    public Action<SummaryV1> OnSummary;
    public Action<PartSummaryRoot> OnPartSummary;
    public Action<WorkerStatusRoot> OnWorkerStatus;
    public Action<RecentAlertsRoot> OnRecentAlerts;

    public TextMeshProUGUI workingContent;
    public TextMeshProUGUI partWorkingContent;
    public TextMeshProUGUI workerRiskAlertContent;
    public TextMeshProUGUI yesterdayRiskContent;

    //private IEnumerator PollLoop()
    //{
    //    do
    //    {
    //        // 필요에 따라 일부만 호출해도 됨
    //        yield return FetchOngoingTasks();
    //        yield return FetchPartSummary();
    //        yield return FetchWorkerStatus();
    //        yield return FetchRecentAlerts(72);
    //        yield return FetchRiskSummary();
    //        yield return FetchZoneSummary();

    //        if (pollIntervalSec <= 0f) break;
    //        yield return new WaitForSeconds(pollIntervalSec);
    //    } while (running);
    //}

    // ─────────────────────────────────────────────────────────────
    // 공통 GET
    private IEnumerator Get(string path, System.Action<string> onOk)
    {
        var url = baseUrl.TrimEnd('/') + "/" + path.TrimStart('/');
        using (var req = UnityWebRequest.Get(url))
        {
            req.timeout = 10;
            req.SetRequestHeader("Accept", "application/json");
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success)
            {
                Debug.LogError($"[HTTP] {path} -> {req.error}, code={req.responseCode}");
                yield break;
            }
            onOk?.Invoke(req.downloadHandler.text);
        }
    }

    // 1) ongoing-tasks <업무 현황표>
    public IEnumerator FetchOngoingTasks()
    {
        yield return Get("/dashboard/ongoing-tasks", json =>
        {
            var data = JsonUtility.FromJson<SummaryV1>(json);
            if (data?.ongoing_tasks == null) { Debug.LogError("[ongoing] null"); return; }

            Debug.Log($"[ongoing] count={data.ongoing_tasks.Length}");
            for (int i = 0; i < Mathf.Min(3, data.ongoing_tasks.Length); i++)
            {
                var t = data.ongoing_tasks[i];
                System.DateTime dt; var ok = System.DateTime.TryParse(t.due_date, out dt);
                Debug.Log($"  · {t.task_name} | {t.part} | due {(ok ? dt.ToString("yyyy-MM-dd HH:mm") : "parse fail")} | {t.progress}% | delayed={t.is_delayed}");

            }

            // UI로 전달
            OnSummary?.Invoke(data);
        });
    }

    // 2) part-summary <파트별 현황>
    public IEnumerator FetchPartSummary()
    {
        yield return Get("/dashboard/part-summary", json =>
        {
            var data = JsonUtility.FromJson<PartSummaryRoot>(json);
            if (data?.part_summary == null) { Debug.LogError("[part] null"); return; }

            Debug.Log($"[part] buckets={data.part_summary.Length}");
            foreach (var p in data.part_summary)
                Debug.Log($"  · {p.part}: tasks={p.task_count}, delay={p.delay_rate:P1}");
            // TODO: UI 바인딩
            OnPartSummary?.Invoke(data);
        });
    }

    // 3) worker-status <작업자별 현황표>
    public IEnumerator FetchWorkerStatus()
    {
        yield return Get("/dashboard/worker-status", json =>
        {
            string s = "";

            var data = JsonUtility.FromJson<WorkerStatusRoot>(json);
            if (data?.worker_status == null) { Debug.LogError("[worker] null"); return; }

            Debug.Log($"[worker] rows={data.worker_status.Length}");
            for (int i = 0; i < Mathf.Min(3, data.worker_status.Length); i++)
            {
                Debug.Log($"  · {data.worker_status[i].worker_id}: total={data.worker_status[i].total_tasks}");
                s += $"  · {data.worker_status[i].worker_id}: total={data.worker_status[i].total_tasks}\n";
            }
            // TODO: UI 바인딩
            //OnWorkerStatus?.Invoke(data);


            workingContent.SetText(s);
        });
    }

    // 4) recent-alerts <최근 알람 내역>
    public IEnumerator FetchRecentAlerts(int hours)
    {
        yield return Get($"/dashboard/recent-alerts?hours={hours}", json =>
        {
            string s = "";

            var data = JsonUtility.FromJson<RecentAlertsRoot>(json);
            if (data?.recent_alerts == null) { Debug.LogError("[alerts] null"); return; }

            Debug.Log($"[alerts] hours={data.hours}, rows={data.recent_alerts.Length}");
            for (int i = 0; i < Mathf.Min(3, data.recent_alerts.Length); i++)
            {
                var r = data.recent_alerts[i];
                Debug.Log($"  · {r.worker_id} | {r.type} x{r.count}");
                s += $"  · {r.worker_id} | {r.type} x{r.count}\n";
            }
            // TODO: UI 바인딩
            //OnRecentAlerts?.Invoke(data);
            workingContent.SetText(s);
        });
    }

    // 5) risk-summary <작업자 경고 상태>
    public IEnumerator FetchRiskSummary()
    {
        yield return Get("/dashboard/risk-summary", json =>
        {
            var data = JsonUtility.FromJson<RiskSummaryRoot>(json);
            if (data?.risk_summary == null) { Debug.LogError("[risk] null"); return; }

            var r = data.risk_summary;
            Debug.Log($"[risk] total_workers={r.total_workers}, ppe%={r.ppe_violation_rate}, roi_cnt={r.roi_violation_count}");
            // TODO: UI 바인딩
        });
    }

    // 6) zone-summary <구역별 현황>
    public IEnumerator FetchZoneSummary()
    {
        yield return Get("/dashboard/zone-summary", json =>
        {
            var data = JsonUtility.FromJson<ZoneSummaryRoot>(json);
            if (data?.zone_summary == null) { Debug.LogError("[zone] null"); return; }

            Debug.Log($"[zone] rows={data.zone_summary.Length}");
            for (int i = 0; i < Mathf.Min(3, data.zone_summary.Length); i++)
            {
                var z = data.zone_summary[i];
                Debug.Log($"  · zone={z.zone_id}, active_workers={z.active_workers}");
            }
            // TODO: UI 바인딩
        });
    }
}
