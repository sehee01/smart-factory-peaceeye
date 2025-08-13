using System;
using UnityEngine;
using UnityEngine.InputSystem;

public class workerCamera : MonoBehaviour
{
    public Transform target;
    public float distance = 10.0f;
    public float xSpeed = 120.0f;
    public float ySpeed = 80.0f;

    float x = 0.0f;
    float y = 0.0f;

    private Vector2 lastMousePosition;
    internal static Camera main;

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        x = angles.y;
        y = angles.x;

        Cursor.lockState = CursorLockMode.None;
    }

    void LateUpdate()
    {
        if (Mouse.current.leftButton.isPressed)
        {
            Vector2 mousePos = Mouse.current.position.ReadValue();
            Vector2 delta = mousePos - lastMousePosition;

            x += delta.x * xSpeed * Time.deltaTime;
            y -= delta.y * ySpeed * Time.deltaTime;
            y = Mathf.Clamp(y, 10, 80); // มฆวั

            lastMousePosition = mousePos;
        }

        Quaternion rotation = Quaternion.Euler(y, x, 0);
        Vector3 position = rotation * new Vector3(0.0f, 0.0f, -distance) + target.position;

        transform.rotation = rotation;
        transform.position = position;
    }

    internal Ray ScreenPointToRay(Vector2 vector2)
    {
        throw new NotImplementedException();
    }
}
