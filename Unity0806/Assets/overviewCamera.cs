using UnityEngine;
using UnityEngine.InputSystem;

public class OverviewCamera : MonoBehaviour
{
    public float rotationSpeed = 10.0f;
    public float zoomSpeed = 200.0f;
    public float minZoom = 500.0f;
    public float maxZoom = 4000.0f;
    public float moveSpeed = 100.0f;

    private Vector2 lastMousePosition;
    private float currentZoom;

    private float fixedHeight;  // 고정 Y 값

    void Start()
    {
        currentZoom = Camera.main.orthographicSize;
        fixedHeight = transform.position.y;
    }

    void Update()
    {
        HandleMouseRotation();
        HandleMouseMovement();
        HandleZoom();
    }

    void HandleMouseRotation()
    {
        if (Mouse.current.rightButton.isPressed)
        {
            Vector2 currentMousePos = Mouse.current.position.ReadValue();

            if (lastMousePosition != Vector2.zero)
            {
                Vector2 delta = currentMousePos - lastMousePosition;

                float rotX = -delta.y * rotationSpeed * Time.deltaTime;
                float rotY = delta.x * rotationSpeed * Time.deltaTime;

                // 회전 중심을 현재 카메라 위치 기준으로 변경
                transform.RotateAround(transform.position, transform.right, rotX);
                transform.RotateAround(transform.position, Vector3.up, rotY);
            }

            lastMousePosition = currentMousePos;
        }
        else if (!Mouse.current.leftButton.isPressed)
        {
            lastMousePosition = Vector2.zero;
        }
    }

    void HandleMouseMovement()
    {
        if (Mouse.current.leftButton.isPressed)
        {
            Vector2 currentMousePos = Mouse.current.position.ReadValue();

            if (lastMousePosition != Vector2.zero)
            {
                Vector2 delta = currentMousePos - lastMousePosition;

                // 카메라 기준 수평 이동 벡터 계산
                Vector3 right = transform.right;
                Vector3 forward = Vector3.Cross(right, Vector3.up).normalized;

                Vector3 move = (-right * delta.x - forward * delta.y) * moveSpeed * Time.deltaTime;

                transform.position += move;

                // 바닥 고정 (높이 유지)
                Vector3 pos = transform.position;
                pos.y = fixedHeight;
                transform.position = pos;
            }

            lastMousePosition = currentMousePos;
        }
        else if (!Mouse.current.rightButton.isPressed)
        {
            lastMousePosition = Vector2.zero;
        }
    }

    void HandleZoom()
    {
        float scroll = Mouse.current.scroll.ReadValue().y;

        float newSize = Camera.main.orthographicSize - scroll * zoomSpeed * Time.deltaTime;
        Camera.main.orthographicSize = Mathf.Clamp(newSize, minZoom, maxZoom);

        /* currentZoom -= scroll * zoomSpeed * Time.deltaTime;
         currentZoom = Mathf.Clamp(currentZoom, minZoom, maxZoom);

         // 카메라 전방 방향으로 줌
         Vector3 forward = transform.forward;
         //Vector3 newPos = transform.position + forward * scroll * zoomSpeed * Time.deltaTime;
         Vector3 pos = transform.position;

         // Y 고정
         pos.y = currentZoom;
         transform.position = pos;
         //newPos.y = currentZoom;
         //transform.position = newPos;*/

        //fixedHeight = currentZoom; // 업데이트된 높이를 유지
    }
}
