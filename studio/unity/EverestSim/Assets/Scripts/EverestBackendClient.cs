using System;
using System.Collections.Concurrent;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

#if UNITY_WEBGL && !UNITY_EDITOR
using System.Runtime.InteropServices;
#else
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;
#endif

namespace EverestSim
{
    public sealed class EverestBackendClient : MonoBehaviour
    {
        private const int MaxMessageBytes = 16 * 1024 * 1024;
        private readonly ConcurrentQueue<string> _incoming = new ConcurrentQueue<string>();

#if UNITY_WEBGL && !UNITY_EDITOR
        [DllImport("__Internal")] private static extern void EverestWebSocketConnect(string url, string receiver);
        [DllImport("__Internal")] private static extern void EverestWebSocketSend(string payload);
        [DllImport("__Internal")] private static extern void EverestWebSocketClose();
        private bool _webConnected;
        private StringBuilder _webChunkBuffer;
        private int _webChunkExpected;
#else
        private readonly SemaphoreSlim _sendLock = new SemaphoreSlim(1, 1);
        private ClientWebSocket _socket;
        private CancellationTokenSource _shutdown;
        private Task _connectionTask;
#endif

        public string Endpoint { get; private set; }
        public bool Connected
        {
            get
            {
#if UNITY_WEBGL && !UNITY_EDITOR
                return _webConnected;
#else
                return _socket != null && _socket.State == WebSocketState.Open;
#endif
            }
        }

        public string ConnectionError { get; private set; }
        public JObject LatestState { get; private set; }
        public JObject LatestEnvironment { get; private set; }
        public JObject LatestSnow { get; private set; }
        public JObject LatestSnowHistory { get; private set; }
        public JObject LatestFault { get; private set; }
        public JObject LatestControlAck { get; private set; }
        public JObject LatestSensors { get; private set; }

        public event Action<JObject> SceneReceived;
        public event Action<JObject> TerrainReceived;
        public event Action<JObject> MacroTerrainReceived;
        public event Action<JObject> FrameReceived;
        public event Action<JObject> SnowReceived;
        public event Action<JObject> SnowHistoryReceived;
        public event Action<JObject> EnvironmentReceived;
        public event Action<JObject> StateReceived;
        public event Action<JObject> FaultReceived;
        public event Action<JObject> ControlAckReceived;
        public event Action<JObject> SensorsReceived;

        private void Awake()
        {
            Endpoint = ResolveEndpoint();
#if !(UNITY_WEBGL && !UNITY_EDITOR)
            _shutdown = new CancellationTokenSource();
#endif
        }

        private void Start()
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            EverestWebSocketConnect(Endpoint, gameObject.name);
#else
            _connectionTask = ConnectionLoopAsync(_shutdown.Token);
#endif
        }

        private void Update()
        {
            var budget = 24;
            while (budget-- > 0 && _incoming.TryDequeue(out var raw))
            {
                try
                {
                    var message = JObject.Parse(raw);
                    var type = message.Value<string>("type") ?? string.Empty;
                    var data = message["data"] as JObject ?? new JObject();
                    Dispatch(type, data);
                }
                catch (Exception exc)
                {
                    Debug.LogWarning($"Everest bridge message parse failed: {exc.Message}");
                }
            }
        }

        private void OnDestroy()
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            EverestWebSocketClose();
            _webConnected = false;
#else
            try { _shutdown?.Cancel(); } catch { }
            try { _socket?.Abort(); } catch { }
            try { _socket?.Dispose(); } catch { }
            _sendLock.Dispose();
            _shutdown?.Dispose();
#endif
        }

        private static string ResolveEndpoint()
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            var absolute = Application.absoluteURL;
            if (!string.IsNullOrWhiteSpace(absolute) && Uri.TryCreate(absolute, UriKind.Absolute, out var pageUri))
            {
                var query = pageUri.Query.TrimStart('?').Split('&');
                foreach (var item in query)
                {
                    var parts = item.Split(new[] { '=' }, 2);
                    if (parts.Length == 2 && string.Equals(parts[0], "backend", StringComparison.OrdinalIgnoreCase))
                    {
                        var decoded = Uri.UnescapeDataString(parts[1]);
                        if (!string.IsNullOrWhiteSpace(decoded)) return decoded;
                    }
                }
                var scheme = string.Equals(pageUri.Scheme, "https", StringComparison.OrdinalIgnoreCase) ? "wss" : "ws";
                return $"{scheme}://{pageUri.Host}:18765";
            }
#else
            var configured = Environment.GetEnvironmentVariable("EVEREST_BACKEND_URL");
            if (!string.IsNullOrWhiteSpace(configured)) return configured;
            var args = Environment.GetCommandLineArgs();
            for (var i = 0; i + 1 < args.Length; ++i)
            {
                if (string.Equals(args[i], "-everestBackend", StringComparison.OrdinalIgnoreCase))
                    return args[i + 1];
            }
#endif
            return "ws://127.0.0.1:18765";
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        public void OnWebSocketOpen(string _)
        {
            _webConnected = true;
            ConnectionError = null;
        }

        public void OnWebSocketMessage(string raw)
        {
            if (!string.IsNullOrEmpty(raw) && Encoding.UTF8.GetByteCount(raw) <= MaxMessageBytes)
                _incoming.Enqueue(raw);
        }

        public void OnWebSocketChunkBegin(string expectedChars)
        {
            if (!int.TryParse(expectedChars, out var expected) || expected <= 0 || expected > MaxMessageBytes)
            {
                _webChunkBuffer = null;
                _webChunkExpected = 0;
                ConnectionError = "Invalid chunked backend message size";
                return;
            }
            _webChunkExpected = expected;
            _webChunkBuffer = new StringBuilder(expected);
        }

        public void OnWebSocketChunk(string chunk)
        {
            if (_webChunkBuffer == null || string.IsNullOrEmpty(chunk)) return;
            if (_webChunkBuffer.Length + chunk.Length > _webChunkExpected)
            {
                _webChunkBuffer = null;
                _webChunkExpected = 0;
                ConnectionError = "Chunked backend message exceeded announced size";
                return;
            }
            _webChunkBuffer.Append(chunk);
        }

        public void OnWebSocketChunkEnd(string _)
        {
            if (_webChunkBuffer == null) return;
            var raw = _webChunkBuffer.ToString();
            _webChunkBuffer = null;
            _webChunkExpected = 0;
            if (!string.IsNullOrEmpty(raw) && Encoding.UTF8.GetByteCount(raw) <= MaxMessageBytes)
                _incoming.Enqueue(raw);
        }

        public void OnWebSocketError(string error)
        {
            _webConnected = false;
            ConnectionError = string.IsNullOrWhiteSpace(error) ? "WebSocket error" : error;
        }

        public void OnWebSocketClose(string reason)
        {
            _webConnected = false;
            ConnectionError = string.IsNullOrWhiteSpace(reason) ? "Backend disconnected; reconnecting" : reason;
        }
#else
        private async Task ConnectionLoopAsync(CancellationToken token)
        {
            while (!token.IsCancellationRequested)
            {
                try
                {
                    ConnectionError = null;
                    var socket = new ClientWebSocket();
                    socket.Options.KeepAliveInterval = TimeSpan.FromSeconds(20);
                    _socket = socket;
                    await socket.ConnectAsync(new Uri(Endpoint), token).ConfigureAwait(false);
                    await ReceiveLoopAsync(socket, token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (token.IsCancellationRequested)
                {
                    return;
                }
                catch (Exception exc)
                {
                    ConnectionError = exc.Message;
                }
                finally
                {
                    try { _socket?.Dispose(); } catch { }
                    _socket = null;
                }

                try { await Task.Delay(1000, token).ConfigureAwait(false); }
                catch (OperationCanceledException) { return; }
            }
        }

        private async Task ReceiveLoopAsync(ClientWebSocket socket, CancellationToken token)
        {
            var chunk = new byte[64 * 1024];
            while (socket.State == WebSocketState.Open && !token.IsCancellationRequested)
            {
                using (var stream = new System.IO.MemoryStream())
                {
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await socket.ReceiveAsync(new ArraySegment<byte>(chunk), token).ConfigureAwait(false);
                        if (result.MessageType == WebSocketMessageType.Close)
                        {
                            await socket.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "renderer closing", token).ConfigureAwait(false);
                            return;
                        }
                        if (result.MessageType != WebSocketMessageType.Text) continue;
                        stream.Write(chunk, 0, result.Count);
                        if (stream.Length > MaxMessageBytes)
                            throw new InvalidOperationException("Everest backend message exceeded 16 MiB");
                    }
                    while (!result.EndOfMessage);

                    if (stream.Length > 0)
                        _incoming.Enqueue(Encoding.UTF8.GetString(stream.ToArray()));
                }
            }
        }
#endif

        private void Dispatch(string type, JObject data)
        {
            switch (type)
            {
                case "scene": SceneReceived?.Invoke(data); break;
                case "terrain": TerrainReceived?.Invoke(data); break;
                case "macro_terrain": MacroTerrainReceived?.Invoke(data); break;
                case "frame": FrameReceived?.Invoke(data); break;
                case "snow":
                    LatestSnow = data;
                    SnowReceived?.Invoke(data);
                    break;
                case "snow_history":
                    LatestSnowHistory = data;
                    SnowHistoryReceived?.Invoke(data);
                    break;
                case "environment":
                    LatestEnvironment = data;
                    EnvironmentReceived?.Invoke(data);
                    break;
                case "state":
                    LatestState = data;
                    StateReceived?.Invoke(data);
                    break;
                case "fault":
                    LatestFault = data;
                    FaultReceived?.Invoke(data);
                    break;
                case "control_ack":
                    LatestControlAck = data;
                    ControlAckReceived?.Invoke(data);
                    break;
                case "sensors":
                    LatestSensors = data;
                    SensorsReceived?.Invoke(data);
                    break;
            }
        }

        public void SendCommand(float forward, float lateral, float yaw) =>
            SendControl("command", new JArray(forward, lateral, yaw));

        public void SendPause(bool paused) => SendControl("pause", JToken.FromObject(paused));
        public void SendReset() => SendControl("reset", JValue.CreateNull());
        public void SendMode(string mode) => SendControl("mode", JToken.FromObject(mode));
        public void SendSurface(string surface) => SendControl("surface", JToken.FromObject(surface));
        public void SendSurfaceFriction(float friction) => SendControl("surface_friction", JToken.FromObject(friction));
        public void SendSimulationSettings(JObject settings) => SendControl("simulation_settings", settings);
        public void SendCheatMode(bool enabled) => SendControl("cheat_mode", JToken.FromObject(enabled));
        public void SendManualForceMode(bool enabled) => SendControl("manual_force_mode", JToken.FromObject(enabled));
        public void SendSnowParameters(JObject parameters) => SendControl("snow_parameters", parameters);
        public void SendWeather(JObject weather) => SendControl("weather", weather);

        public void SendControl(string action, JToken value)
        {
            var payload = new JObject
            {
                ["type"] = "control",
                ["action"] = action,
                ["value"] = value
            }.ToString(Formatting.None);

#if UNITY_WEBGL && !UNITY_EDITOR
            if (Connected) EverestWebSocketSend(payload);
#else
            _ = SendControlAsync(payload, _shutdown.Token);
#endif
        }

#if !(UNITY_WEBGL && !UNITY_EDITOR)
        private async Task SendControlAsync(string payload, CancellationToken token)
        {
            var socket = _socket;
            if (socket == null || socket.State != WebSocketState.Open) return;
            var bytes = Encoding.UTF8.GetBytes(payload);

            await _sendLock.WaitAsync(token).ConfigureAwait(false);
            try
            {
                if (socket.State == WebSocketState.Open)
                {
                    await socket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, token)
                        .ConfigureAwait(false);
                }
            }
            catch (Exception exc) when (!(exc is OperationCanceledException && token.IsCancellationRequested))
            {
                ConnectionError = exc.Message;
            }
            finally
            {
                _sendLock.Release();
            }
        }
#endif
    }
}
