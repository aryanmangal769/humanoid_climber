using UnityEngine;

namespace EverestSim
{
    public enum EverestCameraMode
    {
        Orbit,
        Free
    }

    public sealed class EverestCameraController : MonoBehaviour
    {
        private Camera _camera;
        private EverestRobotRenderer _robot;
        private EverestEditorHud _hud;
        private float _yaw = 205f;
        private float _pitch = 17f;
        private float _distance = 7.2f;
        private float _freeSpeed = 5.5f;
        private Vector3 _focus = new Vector3(0f, 1.0f, 0f);
        private Vector3 _focusOffset;
        private bool _orbitDragging;
        private bool _panDragging;

        public EverestCameraMode Mode { get; private set; } = EverestCameraMode.Orbit;
        public bool FreeMoveModifierHeld => Mode == EverestCameraMode.Free &&
                                            !SceneInputBlocked &&
                                            (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift));
        private bool SceneInputBlocked => _hud != null && _hud.BlocksSceneInput;

        public void Initialize(EverestRobotRenderer robot)
        {
            _robot = robot;
        }

        public void SetEditorHud(EverestEditorHud hud)
        {
            _hud = hud;
        }

        public void SetMode(EverestCameraMode mode)
        {
            if (Mode == mode || _camera == null)
            {
                Mode = mode;
                return;
            }

            Mode = mode;
            if (mode == EverestCameraMode.Free)
            {
                var euler = _camera.transform.eulerAngles;
                _yaw = euler.y;
                _pitch = euler.x > 180f ? euler.x - 360f : euler.x;
            }
        }

        public void ResetCamera()
        {
            Mode = EverestCameraMode.Orbit;
            _yaw = 205f;
            _pitch = 17f;
            _distance = 7.2f;
            _freeSpeed = 5.5f;
            _focusOffset = Vector3.zero;
            _orbitDragging = false;
            _panDragging = false;
            if (_robot != null && _robot.TryGetBodyPosition("pelvis", out var pelvis))
                _focus = pelvis + Vector3.up * 0.75f;
        }

        private void Awake()
        {
            _camera = Camera.main;
            if (_camera == null)
            {
                var go = new GameObject("Everest Camera");
                go.tag = "MainCamera";
                _camera = go.AddComponent<Camera>();
                go.AddComponent<AudioListener>();
            }
            _camera.nearClipPlane = 0.03f;
            _camera.farClipPlane = 25000f;
            _camera.fieldOfView = 58f;
            _camera.clearFlags = CameraClearFlags.Skybox;
            _camera.backgroundColor = new Color(0.12f, 0.20f, 0.30f);
        }

        private void LateUpdate()
        {
            if (_camera == null) return;
            _camera.rect = _hud != null ? _hud.SceneViewportNormalized : new Rect(0f, 0f, 1f, 1f);
            if (Mode == EverestCameraMode.Free)
                UpdateFreeCamera();
            else
                UpdateOrbitCamera();
        }

        private void UpdateOrbitCamera()
        {
            if (_robot != null && _robot.TryGetBodyPosition("pelvis", out var pelvis))
                _focus = Vector3.Lerp(
                    _focus,
                    pelvis + Vector3.up * 0.75f + _focusOffset,
                    1f - Mathf.Exp(-6f * Time.unscaledDeltaTime));

            if (SceneInputBlocked)
            {
                _orbitDragging = false;
                _panDragging = false;
            }
            if (Input.GetMouseButtonDown(0) && !SceneInputBlocked) _orbitDragging = true;
            if (Input.GetMouseButtonUp(0)) _orbitDragging = false;
            if (Input.GetMouseButtonDown(1) && !SceneInputBlocked) _panDragging = true;
            if (Input.GetMouseButtonUp(1)) _panDragging = false;

            if (_orbitDragging)
            {
                _yaw += Input.GetAxis("Mouse X") * 4f;
                _pitch -= Input.GetAxis("Mouse Y") * 3f;
                _pitch = Mathf.Clamp(_pitch, -5f, 72f);
            }

            if (_panDragging)
            {
                var scale = Mathf.Max(0.002f, _distance * 0.0018f);
                var delta = _camera.transform.right * (Input.GetAxis("Mouse X") * scale)
                    + _camera.transform.up * (Input.GetAxis("Mouse Y") * scale);
                _focusOffset -= delta;
                _focus -= delta;
            }

            if (!SceneInputBlocked)
            {
                _distance *= Mathf.Exp(-Input.mouseScrollDelta.y * 0.12f);
                _distance = Mathf.Clamp(_distance, 2.2f, 80f);
            }

            var orbit = Quaternion.Euler(_pitch, _yaw, 0f);
            _camera.transform.position = _focus + orbit * new Vector3(0f, 0f, -_distance);
            _camera.transform.rotation = Quaternion.LookRotation(_focus - _camera.transform.position, Vector3.up);
        }

        private void UpdateFreeCamera()
        {
            if (SceneInputBlocked) _orbitDragging = false;
            if (Input.GetMouseButtonDown(0) && !SceneInputBlocked) _orbitDragging = true;
            if (Input.GetMouseButtonUp(0)) _orbitDragging = false;
            if (!SceneInputBlocked)
            {
                if (_orbitDragging)
                {
                    _yaw += Input.GetAxis("Mouse X") * 3.5f;
                    _pitch -= Input.GetAxis("Mouse Y") * 3.0f;
                    _pitch = Mathf.Clamp(_pitch, -88f, 88f);
                }

                _freeSpeed *= Mathf.Exp(Input.mouseScrollDelta.y * 0.10f);
                _freeSpeed = Mathf.Clamp(_freeSpeed, 1.0f, 80f);
            }

            _camera.transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
            if (SceneInputBlocked) return;

            var arrowsHeld = Input.GetKey(KeyCode.UpArrow)
                || Input.GetKey(KeyCode.DownArrow)
                || Input.GetKey(KeyCode.LeftArrow)
                || Input.GetKey(KeyCode.RightArrow);
            if (!FreeMoveModifierHeld && !arrowsHeld) return;

            var move = Vector3.zero;
            if (Input.GetKey(KeyCode.UpArrow) || Input.GetKey(KeyCode.W)) move += Vector3.forward;
            if (Input.GetKey(KeyCode.DownArrow) || Input.GetKey(KeyCode.S)) move += Vector3.back;
            if (Input.GetKey(KeyCode.LeftArrow) || Input.GetKey(KeyCode.A)) move += Vector3.left;
            if (Input.GetKey(KeyCode.RightArrow) || Input.GetKey(KeyCode.D)) move += Vector3.right;
            if (FreeMoveModifierHeld && Input.GetKey(KeyCode.Q)) move += Vector3.down;
            if (FreeMoveModifierHeld && Input.GetKey(KeyCode.E)) move += Vector3.up;

            if (move.sqrMagnitude > 1f) move.Normalize();
            _camera.transform.position += _camera.transform.TransformDirection(move) * (_freeSpeed * Time.unscaledDeltaTime);
        }
    }
}
