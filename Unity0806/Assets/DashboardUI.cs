using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DashboardUI : MonoBehaviour
{
    [SerializeField] private DashboardClient client;

    [Header("UI References")]
    [SerializeField] private Text tasksCountText;         // UnityEngine.UI.Text
    [SerializeField] private Transform tasksContainer;    // 아이템이 생성될 부모 Transform
    [SerializeField] private GameObject taskItemPrefab;   // Unity UI 프리팹 (Text 포함)

    private void OnEnable()
    {
        if (client != null)
            client.OnSummary += RenderSummary;
    }

    private void OnDisable()
    {
        if (client != null)
            client.OnSummary -= RenderSummary;
    }

    private void RenderSummary(SummaryV1 data)
    {
        // 총 개수 표시
        if (tasksCountText != null)
        {
            tasksCountText.text = $"Tasks: {data.ongoing_tasks.Length}";
        }

        // 기존 아이템 전부 삭제
        foreach (Transform child in tasksContainer)
        {
            Destroy(child.gameObject);
        }

        // 새 아이템 생성
        foreach (var t in data.ongoing_tasks)
        {
            var go = Instantiate(taskItemPrefab, tasksContainer);
            var text = go.GetComponentInChildren<Text>();
            if (text != null)
            {
                text.text = $"{t.task_name} ({t.part}) - {t.progress}%";
            }
        }
    }
}
