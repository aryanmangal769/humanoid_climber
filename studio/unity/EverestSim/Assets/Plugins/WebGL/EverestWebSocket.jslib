mergeInto(LibraryManager.library, {
  EverestWebSocketConnect: function (urlPtr, receiverPtr) {
    var url = UTF8ToString(urlPtr);
    var receiver = UTF8ToString(receiverPtr);

    if (!window.__everestWsState) {
      window.__everestWsState = {
        socket: null,
        closing: false,
        timer: null,
        url: null,
        receiver: null,
        incoming: [],
        pumping: false
      };
    }
    var state = window.__everestWsState;
    state.url = url;
    state.receiver = receiver;
    state.closing = false;
    state.incoming = [];
    state.pumping = false;

    var CHUNK_SIZE = 32768;
    var MAX_MESSAGE_CHARS = 16 * 1024 * 1024;

    function notify(method, message) {
      try { SendMessage(state.receiver, method, message || ''); } catch (e) { console.error(e); }
    }

    function pumpNextMessage() {
      if (state.pumping || state.incoming.length === 0 || state.closing) return;
      var raw = state.incoming.shift();
      if (typeof raw !== 'string') {
        setTimeout(pumpNextMessage, 0);
        return;
      }
      if (raw.length > MAX_MESSAGE_CHARS) {
        notify('OnWebSocketError', 'Backend message exceeded 16 MiB');
        setTimeout(pumpNextMessage, 0);
        return;
      }

      state.pumping = true;
      if (raw.length <= CHUNK_SIZE) {
        notify('OnWebSocketMessage', raw);
        state.pumping = false;
        setTimeout(pumpNextMessage, 0);
        return;
      }

      notify('OnWebSocketChunkBegin', String(raw.length));
      var offset = 0;
      function pumpChunk() {
        if (state.closing) {
          state.pumping = false;
          return;
        }
        if (offset >= raw.length) {
          notify('OnWebSocketChunkEnd', '');
          state.pumping = false;
          setTimeout(pumpNextMessage, 0);
          return;
        }
        var end = Math.min(raw.length, offset + CHUNK_SIZE);
        notify('OnWebSocketChunk', raw.slice(offset, end));
        offset = end;
        // Yield between chunks so Unity can return to the browser main loop.
        setTimeout(pumpChunk, 0);
      }
      setTimeout(pumpChunk, 0);
    }

    function enqueueIncoming(raw) {
      state.incoming.push(raw);
      pumpNextMessage();
    }

    function connect() {
      if (state.closing) return;
      try {
        if (state.socket) {
          try { state.socket.close(); } catch (_) {}
        }
        var ws = new WebSocket(state.url);
        state.socket = ws;
        ws.onopen = function () { notify('OnWebSocketOpen', ''); };
        ws.onmessage = function (event) {
          if (typeof event.data === 'string') enqueueIncoming(event.data);
        };
        ws.onerror = function () { notify('OnWebSocketError', 'WebSocket connection error'); };
        ws.onclose = function (event) {
          notify('OnWebSocketClose', event.reason || 'Backend disconnected; reconnecting');
          if (!state.closing) {
            clearTimeout(state.timer);
            state.timer = setTimeout(connect, 1000);
          }
        };
      } catch (e) {
        notify('OnWebSocketError', String(e));
        if (!state.closing) {
          clearTimeout(state.timer);
          state.timer = setTimeout(connect, 1000);
        }
      }
    }

    connect();
  },

  EverestWebSocketSend: function (payloadPtr) {
    var state = window.__everestWsState;
    if (!state || !state.socket || state.socket.readyState !== WebSocket.OPEN) return;
    state.socket.send(UTF8ToString(payloadPtr));
  },

  EverestWebSocketClose: function () {
    var state = window.__everestWsState;
    if (!state) return;
    state.closing = true;
    state.incoming = [];
    clearTimeout(state.timer);
    if (state.socket) {
      try { state.socket.close(1000, 'Unity renderer closing'); } catch (_) {}
    }
    state.socket = null;
  }
});
