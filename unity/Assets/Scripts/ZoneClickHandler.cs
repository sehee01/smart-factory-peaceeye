using UnityEngine;

public class ZoneClickHandler : MonoBehaviour
{
    public string zoneId;

    void OnMouseDown()
    {
        Debug.Log($"[ZoneClickHandler] 클릭된 Zone: {zoneId}");
        StartCoroutine(ZoneHistoryFetcher.Instance.FetchZoneHistory(zoneId));
    }
}
