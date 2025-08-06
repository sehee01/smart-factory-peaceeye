using UnityEngine;
using UnityEngine.InputSystem;

public class WorkerRaycaster : MonoBehaviour
{
    public Camera mainCamera;             // 메인 카메라 연결
    public WorkerController controller;   // WorkerController 연결
    void Start()
    {
        // 자동으로 mainCamera 설정 (null일 경우만)
        if (mainCamera == null)
            mainCamera = Camera.main;
    }

    void Update() 
    {
        if (Mouse.current.leftButton.wasPressedThisFrame)
        {
            Ray ray = mainCamera.ScreenPointToRay(Mouse.current.position.ReadValue());
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                //Debug.Log(hit.collider.name);
                controller.ShowWorkerInfo(hit.collider.name);
                //controller.ShowWorkerInfo()
                //var handler = hit.collider.GetComponent<WorkerClickHandler>();
                //if (handler != null)
                //{
                //    Debug.Log("Worker 클릭됨 (Raycast): " + handler.workerID);
                //    controller.ShowWorkerInfo(handler.workerID);
                //}
            }
        }
    }
}