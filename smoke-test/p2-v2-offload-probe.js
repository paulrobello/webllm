var __defProp = Object.defineProperty;
var __returnValue = (v) => v;
function __exportSetter(name, newValue) {
  this[name] = __returnValue.bind(null, newValue);
}
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: __exportSetter.bind(all, name)
    });
};
var __esm = (fn, res) => () => (fn && (res = fn(fn = 0)), res);

// smoke-test/webllm-wasm-jsep.js
var exports_webllm_wasm_jsep = {};
__export(exports_webllm_wasm_jsep, {
  default: () => webllm_wasm_jsep_default
});
async function Module(moduleArg = {}) {
  var moduleRtn;
  var Module2 = moduleArg;
  var ENVIRONMENT_IS_WEB = true;
  var ENVIRONMENT_IS_WORKER = false;
  var arguments_ = [];
  var thisProgram = "./this.program";
  var quit_ = (status, toThrow) => {
    throw toThrow;
  };
  var _scriptName = import.meta.url;
  var scriptDirectory = "";
  function locateFile(path) {
    if (Module2["locateFile"]) {
      return Module2["locateFile"](path, scriptDirectory);
    }
    return scriptDirectory + path;
  }
  var readAsync, readBinary;
  if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
    try {
      scriptDirectory = new URL(".", _scriptName).href;
    } catch {}
    {
      readAsync = async (url) => {
        var response = await fetch(url, { credentials: "same-origin" });
        if (response.ok) {
          return response.arrayBuffer();
        }
        throw new Error(response.status + " : " + response.url);
      };
    }
  } else {}
  var out = console.log.bind(console);
  var err = console.error.bind(console);
  var wasmBinary;
  var ABORT = false;
  var EXITSTATUS;
  function assert(condition, text) {
    if (!condition) {
      abort(text);
    }
  }
  var readyPromiseResolve, readyPromiseReject;
  var runtimeInitialized = false;
  function updateMemoryViews() {
    var b = wasmMemory.buffer;
    HEAP8 = new Int8Array(b);
    HEAP16 = new Int16Array(b);
    Module2["HEAPU8"] = HEAPU8 = new Uint8Array(b);
    HEAPU16 = new Uint16Array(b);
    HEAP32 = new Int32Array(b);
    HEAPU32 = new Uint32Array(b);
    Module2["HEAPF32"] = HEAPF32 = new Float32Array(b);
    HEAPF64 = new Float64Array(b);
    HEAP64 = new BigInt64Array(b);
    HEAPU64 = new BigUint64Array(b);
  }
  function preRun() {
    if (Module2["preRun"]) {
      if (typeof Module2["preRun"] == "function")
        Module2["preRun"] = [Module2["preRun"]];
      while (Module2["preRun"].length) {
        addOnPreRun(Module2["preRun"].shift());
      }
    }
    callRuntimeCallbacks(onPreRuns);
  }
  function initRuntime() {
    runtimeInitialized = true;
    if (!Module2["noFSInit"] && !FS.initialized)
      FS.init();
    TTY.init();
    wasmExports["__wasm_call_ctors"]();
    FS.ignorePermissions = false;
  }
  function postRun() {
    if (Module2["postRun"]) {
      if (typeof Module2["postRun"] == "function")
        Module2["postRun"] = [Module2["postRun"]];
      while (Module2["postRun"].length) {
        addOnPostRun(Module2["postRun"].shift());
      }
    }
    callRuntimeCallbacks(onPostRuns);
  }
  function abort(what) {
    Module2["onAbort"]?.(what);
    what = `Aborted(${what})`;
    err(what);
    ABORT = true;
    what += ". Build with -sASSERTIONS for more info.";
    if (runtimeInitialized) {
      ___trap();
    }
    var e = new WebAssembly.RuntimeError(what);
    readyPromiseReject?.(e);
    throw e;
  }
  var wasmBinaryFile;
  function findWasmBinary() {
    if (Module2["locateFile"]) {
      return locateFile("webllm-wasm-jsep.wasm");
    }
    return new URL("webllm-wasm-jsep.wasm", import.meta.url).href;
  }
  function getBinarySync(file) {
    if (file == wasmBinaryFile && wasmBinary) {
      return new Uint8Array(wasmBinary);
    }
    if (readBinary) {
      return readBinary(file);
    }
    throw "both async and sync fetching of the wasm failed";
  }
  async function getWasmBinary(binaryFile) {
    if (!wasmBinary) {
      try {
        var response = await readAsync(binaryFile);
        return new Uint8Array(response);
      } catch {}
    }
    return getBinarySync(binaryFile);
  }
  async function instantiateArrayBuffer(binaryFile, imports) {
    try {
      var binary = await getWasmBinary(binaryFile);
      var instance = await WebAssembly.instantiate(binary, imports);
      return instance;
    } catch (reason) {
      err(`failed to asynchronously prepare wasm: ${reason}`);
      abort(reason);
    }
  }
  async function instantiateAsync(binary, binaryFile, imports) {
    if (!binary) {
      try {
        var response = fetch(binaryFile, { credentials: "same-origin" });
        var instantiationResult = await WebAssembly.instantiateStreaming(response, imports);
        return instantiationResult;
      } catch (reason) {
        err(`wasm streaming compile failed: ${reason}`);
        err("falling back to ArrayBuffer instantiation");
      }
    }
    return instantiateArrayBuffer(binaryFile, imports);
  }
  function getWasmImports() {
    Asyncify.instrumentWasmImports(wasmImports);
    var imports = { env: wasmImports, wasi_snapshot_preview1: wasmImports };
    return imports;
  }
  async function createWasm() {
    function receiveInstance(instance, module) {
      wasmExports = instance.exports;
      wasmExports = Asyncify.instrumentWasmExports(wasmExports);
      wasmExports = applySignatureConversions(wasmExports);
      assignWasmExports(wasmExports);
      updateMemoryViews();
      return wasmExports;
    }
    function receiveInstantiationResult(result2) {
      return receiveInstance(result2["instance"]);
    }
    var info = getWasmImports();
    if (Module2["instantiateWasm"]) {
      return new Promise((resolve, reject) => {
        Module2["instantiateWasm"](info, (inst, mod) => {
          resolve(receiveInstance(inst, mod));
        });
      });
    }
    wasmBinaryFile ??= findWasmBinary();
    var result = await instantiateAsync(wasmBinary, wasmBinaryFile, info);
    var exports = receiveInstantiationResult(result);
    return exports;
  }

  class ExitStatus {
    name = "ExitStatus";
    constructor(status) {
      this.message = `Program terminated with exit(${status})`;
      this.status = status;
    }
  }
  var HEAP16;
  var HEAP32;
  var HEAP64;
  var HEAP8;
  var HEAPF32;
  var HEAPF64;
  var HEAPU16;
  var HEAPU32;
  var HEAPU64;
  var HEAPU8;
  var callRuntimeCallbacks = (callbacks) => {
    while (callbacks.length > 0) {
      callbacks.shift()(Module2);
    }
  };
  var onPostRuns = [];
  var addOnPostRun = (cb) => onPostRuns.push(cb);
  var onPreRuns = [];
  var addOnPreRun = (cb) => onPreRuns.push(cb);
  var noExitRuntime = true;
  var syscallGetVarargI = () => {
    var ret = HEAP32[+SYSCALLS.varargs >>> 2 >>> 0];
    SYSCALLS.varargs += 4;
    return ret;
  };
  var syscallGetVarargP = syscallGetVarargI;
  var PATH = { isAbs: (path) => path.charAt(0) === "/", splitPath: (filename) => {
    var splitPathRe = /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
    return splitPathRe.exec(filename).slice(1);
  }, normalizeArray: (parts, allowAboveRoot) => {
    var up = 0;
    for (var i = parts.length - 1;i >= 0; i--) {
      var last = parts[i];
      if (last === ".") {
        parts.splice(i, 1);
      } else if (last === "..") {
        parts.splice(i, 1);
        up++;
      } else if (up) {
        parts.splice(i, 1);
        up--;
      }
    }
    if (allowAboveRoot) {
      for (;up; up--) {
        parts.unshift("..");
      }
    }
    return parts;
  }, normalize: (path) => {
    var isAbsolute = PATH.isAbs(path), trailingSlash = path.slice(-1) === "/";
    path = PATH.normalizeArray(path.split("/").filter((p) => !!p), !isAbsolute).join("/");
    if (!path && !isAbsolute) {
      path = ".";
    }
    if (path && trailingSlash) {
      path += "/";
    }
    return (isAbsolute ? "/" : "") + path;
  }, dirname: (path) => {
    var result = PATH.splitPath(path), root = result[0], dir = result[1];
    if (!root && !dir) {
      return ".";
    }
    if (dir) {
      dir = dir.slice(0, -1);
    }
    return root + dir;
  }, basename: (path) => path && path.match(/([^\/]+|\/)\/*$/)[1], join: (...paths) => PATH.normalize(paths.join("/")), join2: (l, r) => PATH.normalize(l + "/" + r) };
  var initRandomFill = () => (view) => (crypto.getRandomValues(view), 0);
  var randomFill = (view) => (randomFill = initRandomFill())(view);
  var PATH_FS = { resolve: (...args) => {
    var resolvedPath = "", resolvedAbsolute = false;
    for (var i = args.length - 1;i >= -1 && !resolvedAbsolute; i--) {
      var path = i >= 0 ? args[i] : FS.cwd();
      if (typeof path != "string") {
        throw new TypeError("Arguments to path.resolve must be strings");
      } else if (!path) {
        return "";
      }
      resolvedPath = path + "/" + resolvedPath;
      resolvedAbsolute = PATH.isAbs(path);
    }
    resolvedPath = PATH.normalizeArray(resolvedPath.split("/").filter((p) => !!p), !resolvedAbsolute).join("/");
    return (resolvedAbsolute ? "/" : "") + resolvedPath || ".";
  }, relative: (from, to) => {
    from = PATH_FS.resolve(from).slice(1);
    to = PATH_FS.resolve(to).slice(1);
    function trim(arr) {
      var start = 0;
      for (;start < arr.length; start++) {
        if (arr[start] !== "")
          break;
      }
      var end = arr.length - 1;
      for (;end >= 0; end--) {
        if (arr[end] !== "")
          break;
      }
      if (start > end)
        return [];
      return arr.slice(start, end - start + 1);
    }
    var fromParts = trim(from.split("/"));
    var toParts = trim(to.split("/"));
    var length = Math.min(fromParts.length, toParts.length);
    var samePartsLength = length;
    for (var i = 0;i < length; i++) {
      if (fromParts[i] !== toParts[i]) {
        samePartsLength = i;
        break;
      }
    }
    var outputParts = [];
    for (var i = samePartsLength;i < fromParts.length; i++) {
      outputParts.push("..");
    }
    outputParts = outputParts.concat(toParts.slice(samePartsLength));
    return outputParts.join("/");
  } };
  var UTF8Decoder = globalThis.TextDecoder && new TextDecoder;
  var findStringEnd = (heapOrArray, idx, maxBytesToRead, ignoreNul) => {
    var maxIdx = idx + maxBytesToRead;
    if (ignoreNul)
      return maxIdx;
    while (heapOrArray[idx] && !(idx >= maxIdx))
      ++idx;
    return idx;
  };
  var UTF8ArrayToString = (heapOrArray, idx = 0, maxBytesToRead, ignoreNul) => {
    idx >>>= 0;
    var endPtr = findStringEnd(heapOrArray, idx, maxBytesToRead, ignoreNul);
    if (endPtr - idx > 16 && heapOrArray.buffer && UTF8Decoder) {
      return UTF8Decoder.decode(heapOrArray.subarray(idx, endPtr));
    }
    var str = "";
    while (idx < endPtr) {
      var u0 = heapOrArray[idx++];
      if (!(u0 & 128)) {
        str += String.fromCharCode(u0);
        continue;
      }
      var u1 = heapOrArray[idx++] & 63;
      if ((u0 & 224) == 192) {
        str += String.fromCharCode((u0 & 31) << 6 | u1);
        continue;
      }
      var u2 = heapOrArray[idx++] & 63;
      if ((u0 & 240) == 224) {
        u0 = (u0 & 15) << 12 | u1 << 6 | u2;
      } else {
        u0 = (u0 & 7) << 18 | u1 << 12 | u2 << 6 | heapOrArray[idx++] & 63;
      }
      if (u0 < 65536) {
        str += String.fromCharCode(u0);
      } else {
        var ch = u0 - 65536;
        str += String.fromCharCode(55296 | ch >> 10, 56320 | ch & 1023);
      }
    }
    return str;
  };
  var FS_stdin_getChar_buffer = [];
  var lengthBytesUTF8 = (str) => {
    var len = 0;
    for (var i = 0;i < str.length; ++i) {
      var c = str.charCodeAt(i);
      if (c <= 127) {
        len++;
      } else if (c <= 2047) {
        len += 2;
      } else if (c >= 55296 && c <= 57343) {
        len += 4;
        ++i;
      } else {
        len += 3;
      }
    }
    return len;
  };
  var stringToUTF8Array = (str, heap, outIdx, maxBytesToWrite) => {
    outIdx >>>= 0;
    if (!(maxBytesToWrite > 0))
      return 0;
    var startIdx = outIdx;
    var endIdx = outIdx + maxBytesToWrite - 1;
    for (var i = 0;i < str.length; ++i) {
      var u = str.codePointAt(i);
      if (u <= 127) {
        if (outIdx >= endIdx)
          break;
        heap[outIdx++ >>> 0] = u;
      } else if (u <= 2047) {
        if (outIdx + 1 >= endIdx)
          break;
        heap[outIdx++ >>> 0] = 192 | u >> 6;
        heap[outIdx++ >>> 0] = 128 | u & 63;
      } else if (u <= 65535) {
        if (outIdx + 2 >= endIdx)
          break;
        heap[outIdx++ >>> 0] = 224 | u >> 12;
        heap[outIdx++ >>> 0] = 128 | u >> 6 & 63;
        heap[outIdx++ >>> 0] = 128 | u & 63;
      } else {
        if (outIdx + 3 >= endIdx)
          break;
        heap[outIdx++ >>> 0] = 240 | u >> 18;
        heap[outIdx++ >>> 0] = 128 | u >> 12 & 63;
        heap[outIdx++ >>> 0] = 128 | u >> 6 & 63;
        heap[outIdx++ >>> 0] = 128 | u & 63;
        i++;
      }
    }
    heap[outIdx >>> 0] = 0;
    return outIdx - startIdx;
  };
  var intArrayFromString = (stringy, dontAddNull, length) => {
    var len = length > 0 ? length : lengthBytesUTF8(stringy) + 1;
    var u8array = new Array(len);
    var numBytesWritten = stringToUTF8Array(stringy, u8array, 0, u8array.length);
    if (dontAddNull)
      u8array.length = numBytesWritten;
    return u8array;
  };
  var FS_stdin_getChar = () => {
    if (!FS_stdin_getChar_buffer.length) {
      var result = null;
      if (globalThis.window?.prompt) {
        result = window.prompt("Input: ");
        if (result !== null) {
          result += `
`;
        }
      } else {}
      if (!result) {
        return null;
      }
      FS_stdin_getChar_buffer = intArrayFromString(result, true);
    }
    return FS_stdin_getChar_buffer.shift();
  };
  var TTY = { ttys: [], init() {}, shutdown() {}, register(dev, ops) {
    TTY.ttys[dev] = { input: [], output: [], ops };
    FS.registerDevice(dev, TTY.stream_ops);
  }, stream_ops: { open(stream) {
    var tty = TTY.ttys[stream.node.rdev];
    if (!tty) {
      throw new FS.ErrnoError(43);
    }
    stream.tty = tty;
    stream.seekable = false;
  }, close(stream) {
    stream.tty.ops.fsync(stream.tty);
  }, fsync(stream) {
    stream.tty.ops.fsync(stream.tty);
  }, read(stream, buffer, offset, length, pos) {
    if (!stream.tty || !stream.tty.ops.get_char) {
      throw new FS.ErrnoError(60);
    }
    var bytesRead = 0;
    for (var i = 0;i < length; i++) {
      var result;
      try {
        result = stream.tty.ops.get_char(stream.tty);
      } catch (e) {
        throw new FS.ErrnoError(29);
      }
      if (result === undefined && bytesRead === 0) {
        throw new FS.ErrnoError(6);
      }
      if (result === null || result === undefined)
        break;
      bytesRead++;
      buffer[offset + i] = result;
    }
    if (bytesRead) {
      stream.node.atime = Date.now();
    }
    return bytesRead;
  }, write(stream, buffer, offset, length, pos) {
    if (!stream.tty || !stream.tty.ops.put_char) {
      throw new FS.ErrnoError(60);
    }
    try {
      for (var i = 0;i < length; i++) {
        stream.tty.ops.put_char(stream.tty, buffer[offset + i]);
      }
    } catch (e) {
      throw new FS.ErrnoError(29);
    }
    if (length) {
      stream.node.mtime = stream.node.ctime = Date.now();
    }
    return i;
  } }, default_tty_ops: { get_char(tty) {
    return FS_stdin_getChar();
  }, put_char(tty, val) {
    if (val === null || val === 10) {
      out(UTF8ArrayToString(tty.output));
      tty.output = [];
    } else {
      if (val != 0)
        tty.output.push(val);
    }
  }, fsync(tty) {
    if (tty.output?.length > 0) {
      out(UTF8ArrayToString(tty.output));
      tty.output = [];
    }
  }, ioctl_tcgets(tty) {
    return { c_iflag: 25856, c_oflag: 5, c_cflag: 191, c_lflag: 35387, c_cc: [3, 28, 127, 21, 4, 0, 1, 0, 17, 19, 26, 0, 18, 15, 23, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] };
  }, ioctl_tcsets(tty, optional_actions, data) {
    return 0;
  }, ioctl_tiocgwinsz(tty) {
    return [24, 80];
  } }, default_tty1_ops: { put_char(tty, val) {
    if (val === null || val === 10) {
      err(UTF8ArrayToString(tty.output));
      tty.output = [];
    } else {
      if (val != 0)
        tty.output.push(val);
    }
  }, fsync(tty) {
    if (tty.output?.length > 0) {
      err(UTF8ArrayToString(tty.output));
      tty.output = [];
    }
  } } };
  var zeroMemory = (ptr, size) => HEAPU8.fill(0, ptr, ptr + size);
  var alignMemory = (size, alignment) => Math.ceil(size / alignment) * alignment;
  var mmapAlloc = (size) => {
    size = alignMemory(size, 65536);
    var ptr = _emscripten_builtin_memalign(65536, size);
    if (ptr)
      zeroMemory(ptr, size);
    return ptr;
  };
  var MEMFS = { ops_table: null, mount(mount) {
    return MEMFS.createNode(null, "/", 16895, 0);
  }, createNode(parent, name, mode, dev) {
    if (FS.isBlkdev(mode) || FS.isFIFO(mode)) {
      throw new FS.ErrnoError(63);
    }
    MEMFS.ops_table ||= { dir: { node: { getattr: MEMFS.node_ops.getattr, setattr: MEMFS.node_ops.setattr, lookup: MEMFS.node_ops.lookup, mknod: MEMFS.node_ops.mknod, rename: MEMFS.node_ops.rename, unlink: MEMFS.node_ops.unlink, rmdir: MEMFS.node_ops.rmdir, readdir: MEMFS.node_ops.readdir, symlink: MEMFS.node_ops.symlink }, stream: { llseek: MEMFS.stream_ops.llseek } }, file: { node: { getattr: MEMFS.node_ops.getattr, setattr: MEMFS.node_ops.setattr }, stream: { llseek: MEMFS.stream_ops.llseek, read: MEMFS.stream_ops.read, write: MEMFS.stream_ops.write, mmap: MEMFS.stream_ops.mmap, msync: MEMFS.stream_ops.msync } }, link: { node: { getattr: MEMFS.node_ops.getattr, setattr: MEMFS.node_ops.setattr, readlink: MEMFS.node_ops.readlink }, stream: {} }, chrdev: { node: { getattr: MEMFS.node_ops.getattr, setattr: MEMFS.node_ops.setattr }, stream: FS.chrdev_stream_ops } };
    var node = FS.createNode(parent, name, mode, dev);
    if (FS.isDir(node.mode)) {
      node.node_ops = MEMFS.ops_table.dir.node;
      node.stream_ops = MEMFS.ops_table.dir.stream;
      node.contents = {};
    } else if (FS.isFile(node.mode)) {
      node.node_ops = MEMFS.ops_table.file.node;
      node.stream_ops = MEMFS.ops_table.file.stream;
      node.usedBytes = 0;
      node.contents = MEMFS.emptyFileContents ??= new Uint8Array(0);
    } else if (FS.isLink(node.mode)) {
      node.node_ops = MEMFS.ops_table.link.node;
      node.stream_ops = MEMFS.ops_table.link.stream;
    } else if (FS.isChrdev(node.mode)) {
      node.node_ops = MEMFS.ops_table.chrdev.node;
      node.stream_ops = MEMFS.ops_table.chrdev.stream;
    }
    node.atime = node.mtime = node.ctime = Date.now();
    if (parent) {
      parent.contents[name] = node;
      parent.atime = parent.mtime = parent.ctime = node.atime;
    }
    return node;
  }, getFileDataAsTypedArray(node) {
    return node.contents.subarray(0, node.usedBytes);
  }, expandFileStorage(node, newCapacity) {
    var prevCapacity = node.contents.length;
    if (prevCapacity >= newCapacity)
      return;
    var CAPACITY_DOUBLING_MAX = 1024 * 1024;
    newCapacity = Math.max(newCapacity, prevCapacity * (prevCapacity < CAPACITY_DOUBLING_MAX ? 2 : 1.125) >>> 0);
    if (prevCapacity)
      newCapacity = Math.max(newCapacity, 256);
    var oldContents = MEMFS.getFileDataAsTypedArray(node);
    node.contents = new Uint8Array(newCapacity);
    node.contents.set(oldContents);
  }, resizeFileStorage(node, newSize) {
    if (node.usedBytes == newSize)
      return;
    var oldContents = node.contents;
    node.contents = new Uint8Array(newSize);
    node.contents.set(oldContents.subarray(0, Math.min(newSize, node.usedBytes)));
    node.usedBytes = newSize;
  }, node_ops: { getattr(node) {
    var attr = {};
    attr.dev = FS.isChrdev(node.mode) ? node.id : 1;
    attr.ino = node.id;
    attr.mode = node.mode;
    attr.nlink = 1;
    attr.uid = 0;
    attr.gid = 0;
    attr.rdev = node.rdev;
    if (FS.isDir(node.mode)) {
      attr.size = 4096;
    } else if (FS.isFile(node.mode)) {
      attr.size = node.usedBytes;
    } else if (FS.isLink(node.mode)) {
      attr.size = node.link.length;
    } else {
      attr.size = 0;
    }
    attr.atime = new Date(node.atime);
    attr.mtime = new Date(node.mtime);
    attr.ctime = new Date(node.ctime);
    attr.blksize = 4096;
    attr.blocks = Math.ceil(attr.size / attr.blksize);
    return attr;
  }, setattr(node, attr) {
    for (const key of ["mode", "atime", "mtime", "ctime"]) {
      if (attr[key] != null) {
        node[key] = attr[key];
      }
    }
    if (attr.size !== undefined) {
      MEMFS.resizeFileStorage(node, attr.size);
    }
  }, lookup(parent, name) {
    if (!MEMFS.doesNotExistError) {
      MEMFS.doesNotExistError = new FS.ErrnoError(44);
      MEMFS.doesNotExistError.stack = "<generic error, no stack>";
    }
    throw MEMFS.doesNotExistError;
  }, mknod(parent, name, mode, dev) {
    return MEMFS.createNode(parent, name, mode, dev);
  }, rename(old_node, new_dir, new_name) {
    var new_node;
    try {
      new_node = FS.lookupNode(new_dir, new_name);
    } catch (e) {}
    if (new_node) {
      if (FS.isDir(old_node.mode)) {
        for (var i in new_node.contents) {
          throw new FS.ErrnoError(55);
        }
      }
      FS.hashRemoveNode(new_node);
    }
    delete old_node.parent.contents[old_node.name];
    new_dir.contents[new_name] = old_node;
    old_node.name = new_name;
    new_dir.ctime = new_dir.mtime = old_node.parent.ctime = old_node.parent.mtime = Date.now();
  }, unlink(parent, name) {
    delete parent.contents[name];
    parent.ctime = parent.mtime = Date.now();
  }, rmdir(parent, name) {
    var node = FS.lookupNode(parent, name);
    for (var i in node.contents) {
      throw new FS.ErrnoError(55);
    }
    delete parent.contents[name];
    parent.ctime = parent.mtime = Date.now();
  }, readdir(node) {
    return [".", "..", ...Object.keys(node.contents)];
  }, symlink(parent, newname, oldpath) {
    var node = MEMFS.createNode(parent, newname, 511 | 40960, 0);
    node.link = oldpath;
    return node;
  }, readlink(node) {
    if (!FS.isLink(node.mode)) {
      throw new FS.ErrnoError(28);
    }
    return node.link;
  } }, stream_ops: { read(stream, buffer, offset, length, position) {
    var contents = stream.node.contents;
    if (position >= stream.node.usedBytes)
      return 0;
    var size = Math.min(stream.node.usedBytes - position, length);
    buffer.set(contents.subarray(position, position + size), offset);
    return size;
  }, write(stream, buffer, offset, length, position, canOwn) {
    if (buffer.buffer === HEAP8.buffer) {
      canOwn = false;
    }
    if (!length)
      return 0;
    var node = stream.node;
    node.mtime = node.ctime = Date.now();
    if (canOwn) {
      node.contents = buffer.subarray(offset, offset + length);
      node.usedBytes = length;
    } else if (node.usedBytes === 0 && position === 0) {
      node.contents = buffer.slice(offset, offset + length);
      node.usedBytes = length;
    } else {
      MEMFS.expandFileStorage(node, position + length);
      node.contents.set(buffer.subarray(offset, offset + length), position);
      node.usedBytes = Math.max(node.usedBytes, position + length);
    }
    return length;
  }, llseek(stream, offset, whence) {
    var position = offset;
    if (whence === 1) {
      position += stream.position;
    } else if (whence === 2) {
      if (FS.isFile(stream.node.mode)) {
        position += stream.node.usedBytes;
      }
    }
    if (position < 0) {
      throw new FS.ErrnoError(28);
    }
    return position;
  }, mmap(stream, length, position, prot, flags) {
    if (!FS.isFile(stream.node.mode)) {
      throw new FS.ErrnoError(43);
    }
    var ptr;
    var allocated;
    var contents = stream.node.contents;
    if (!(flags & 2) && contents.buffer === HEAP8.buffer) {
      allocated = false;
      ptr = contents.byteOffset;
    } else {
      allocated = true;
      ptr = mmapAlloc(length);
      if (!ptr) {
        throw new FS.ErrnoError(48);
      }
      if (contents) {
        if (position > 0 || position + length < contents.length) {
          if (contents.subarray) {
            contents = contents.subarray(position, position + length);
          } else {
            contents = Array.prototype.slice.call(contents, position, position + length);
          }
        }
        HEAP8.set(contents, ptr >>> 0);
      }
    }
    return { ptr, allocated };
  }, msync(stream, buffer, offset, length, mmapFlags) {
    MEMFS.stream_ops.write(stream, buffer, 0, length, offset, false);
    return 0;
  } } };
  var FS_modeStringToFlags = (str) => {
    if (typeof str != "string")
      return str;
    var flagModes = { r: 0, "r+": 2, w: 512 | 64 | 1, "w+": 512 | 64 | 2, a: 1024 | 64 | 1, "a+": 1024 | 64 | 2 };
    var flags = flagModes[str];
    if (typeof flags == "undefined") {
      throw new Error(`Unknown file open mode: ${str}`);
    }
    return flags;
  };
  var FS_fileDataToTypedArray = (data) => {
    if (typeof data == "string") {
      data = intArrayFromString(data, true);
    }
    if (!data.subarray) {
      data = new Uint8Array(data);
    }
    return data;
  };
  var FS_getMode = (canRead, canWrite) => {
    var mode = 0;
    if (canRead)
      mode |= 292 | 73;
    if (canWrite)
      mode |= 146;
    return mode;
  };
  var asyncLoad = async (url) => {
    var arrayBuffer = await readAsync(url);
    return new Uint8Array(arrayBuffer);
  };
  var FS_createDataFile = (...args) => FS.createDataFile(...args);
  var getUniqueRunDependency = (id) => id;
  var runDependencies = 0;
  var dependenciesFulfilled = null;
  var removeRunDependency = (id) => {
    runDependencies--;
    Module2["monitorRunDependencies"]?.(runDependencies);
    if (runDependencies == 0) {
      if (dependenciesFulfilled) {
        var callback = dependenciesFulfilled;
        dependenciesFulfilled = null;
        callback();
      }
    }
  };
  var addRunDependency = (id) => {
    runDependencies++;
    Module2["monitorRunDependencies"]?.(runDependencies);
  };
  var preloadPlugins = [];
  var FS_handledByPreloadPlugin = async (byteArray, fullname) => {
    if (typeof Browser != "undefined")
      Browser.init();
    for (var plugin of preloadPlugins) {
      if (plugin["canHandle"](fullname)) {
        return plugin["handle"](byteArray, fullname);
      }
    }
    return byteArray;
  };
  var FS_preloadFile = async (parent, name, url, canRead, canWrite, dontCreateFile, canOwn, preFinish) => {
    var fullname = name ? PATH_FS.resolve(PATH.join2(parent, name)) : parent;
    var dep = getUniqueRunDependency(`cp ${fullname}`);
    addRunDependency(dep);
    try {
      var byteArray = url;
      if (typeof url == "string") {
        byteArray = await asyncLoad(url);
      }
      byteArray = await FS_handledByPreloadPlugin(byteArray, fullname);
      preFinish?.();
      if (!dontCreateFile) {
        FS_createDataFile(parent, name, byteArray, canRead, canWrite, canOwn);
      }
    } finally {
      removeRunDependency(dep);
    }
  };
  var FS_createPreloadedFile = (parent, name, url, canRead, canWrite, onload, onerror, dontCreateFile, canOwn, preFinish) => {
    FS_preloadFile(parent, name, url, canRead, canWrite, dontCreateFile, canOwn, preFinish).then(onload).catch(onerror);
  };
  var FS = { root: null, mounts: [], devices: {}, streams: [], nextInode: 1, nameTable: null, currentPath: "/", initialized: false, ignorePermissions: true, filesystems: null, syncFSRequests: 0, ErrnoError: class {
    name = "ErrnoError";
    constructor(errno) {
      this.errno = errno;
    }
  }, FSStream: class {
    shared = {};
    get object() {
      return this.node;
    }
    set object(val) {
      this.node = val;
    }
    get isRead() {
      return (this.flags & 2097155) !== 1;
    }
    get isWrite() {
      return (this.flags & 2097155) !== 0;
    }
    get isAppend() {
      return this.flags & 1024;
    }
    get flags() {
      return this.shared.flags;
    }
    set flags(val) {
      this.shared.flags = val;
    }
    get position() {
      return this.shared.position;
    }
    set position(val) {
      this.shared.position = val;
    }
  }, FSNode: class {
    node_ops = {};
    stream_ops = {};
    readMode = 292 | 73;
    writeMode = 146;
    mounted = null;
    constructor(parent, name, mode, rdev) {
      if (!parent) {
        parent = this;
      }
      this.parent = parent;
      this.mount = parent.mount;
      this.id = FS.nextInode++;
      this.name = name;
      this.mode = mode;
      this.rdev = rdev;
      this.atime = this.mtime = this.ctime = Date.now();
    }
    get read() {
      return (this.mode & this.readMode) === this.readMode;
    }
    set read(val) {
      val ? this.mode |= this.readMode : this.mode &= ~this.readMode;
    }
    get write() {
      return (this.mode & this.writeMode) === this.writeMode;
    }
    set write(val) {
      val ? this.mode |= this.writeMode : this.mode &= ~this.writeMode;
    }
    get isFolder() {
      return FS.isDir(this.mode);
    }
    get isDevice() {
      return FS.isChrdev(this.mode);
    }
  }, lookupPath(path, opts = {}) {
    if (!path) {
      throw new FS.ErrnoError(44);
    }
    opts.follow_mount ??= true;
    if (!PATH.isAbs(path)) {
      path = FS.cwd() + "/" + path;
    }
    linkloop:
      for (var nlinks = 0;nlinks < 40; nlinks++) {
        var parts = path.split("/").filter((p) => !!p);
        var current = FS.root;
        var current_path = "/";
        for (var i = 0;i < parts.length; i++) {
          var islast = i === parts.length - 1;
          if (islast && opts.parent) {
            break;
          }
          if (parts[i] === ".") {
            continue;
          }
          if (parts[i] === "..") {
            current_path = PATH.dirname(current_path);
            if (FS.isRoot(current)) {
              path = current_path + "/" + parts.slice(i + 1).join("/");
              nlinks--;
              continue linkloop;
            } else {
              current = current.parent;
            }
            continue;
          }
          current_path = PATH.join2(current_path, parts[i]);
          try {
            current = FS.lookupNode(current, parts[i]);
          } catch (e) {
            if (e?.errno === 44 && islast && opts.noent_okay) {
              return { path: current_path };
            }
            throw e;
          }
          if (FS.isMountpoint(current) && (!islast || opts.follow_mount)) {
            current = current.mounted.root;
          }
          if (FS.isLink(current.mode) && (!islast || opts.follow)) {
            if (!current.node_ops.readlink) {
              throw new FS.ErrnoError(52);
            }
            var link = current.node_ops.readlink(current);
            if (!PATH.isAbs(link)) {
              link = PATH.dirname(current_path) + "/" + link;
            }
            path = link + "/" + parts.slice(i + 1).join("/");
            continue linkloop;
          }
        }
        return { path: current_path, node: current };
      }
    throw new FS.ErrnoError(32);
  }, getPath(node) {
    var path;
    while (true) {
      if (FS.isRoot(node)) {
        var mount = node.mount.mountpoint;
        if (!path)
          return mount;
        return mount[mount.length - 1] !== "/" ? `${mount}/${path}` : mount + path;
      }
      path = path ? `${node.name}/${path}` : node.name;
      node = node.parent;
    }
  }, hashName(parentid, name) {
    var hash = 0;
    for (var i = 0;i < name.length; i++) {
      hash = (hash << 5) - hash + name.charCodeAt(i) | 0;
    }
    return (parentid + hash >>> 0) % FS.nameTable.length;
  }, hashAddNode(node) {
    var hash = FS.hashName(node.parent.id, node.name);
    node.name_next = FS.nameTable[hash];
    FS.nameTable[hash] = node;
  }, hashRemoveNode(node) {
    var hash = FS.hashName(node.parent.id, node.name);
    if (FS.nameTable[hash] === node) {
      FS.nameTable[hash] = node.name_next;
    } else {
      var current = FS.nameTable[hash];
      while (current) {
        if (current.name_next === node) {
          current.name_next = node.name_next;
          break;
        }
        current = current.name_next;
      }
    }
  }, lookupNode(parent, name) {
    var errCode = FS.mayLookup(parent);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    var hash = FS.hashName(parent.id, name);
    for (var node = FS.nameTable[hash];node; node = node.name_next) {
      var nodeName = node.name;
      if (node.parent.id === parent.id && nodeName === name) {
        return node;
      }
    }
    return FS.lookup(parent, name);
  }, createNode(parent, name, mode, rdev) {
    var node = new FS.FSNode(parent, name, mode, rdev);
    FS.hashAddNode(node);
    return node;
  }, destroyNode(node) {
    FS.hashRemoveNode(node);
  }, isRoot(node) {
    return node === node.parent;
  }, isMountpoint(node) {
    return !!node.mounted;
  }, isFile(mode) {
    return (mode & 61440) === 32768;
  }, isDir(mode) {
    return (mode & 61440) === 16384;
  }, isLink(mode) {
    return (mode & 61440) === 40960;
  }, isChrdev(mode) {
    return (mode & 61440) === 8192;
  }, isBlkdev(mode) {
    return (mode & 61440) === 24576;
  }, isFIFO(mode) {
    return (mode & 61440) === 4096;
  }, isSocket(mode) {
    return (mode & 49152) === 49152;
  }, flagsToPermissionString(flag) {
    var perms = ["r", "w", "rw"][flag & 3];
    if (flag & 512) {
      perms += "w";
    }
    return perms;
  }, nodePermissions(node, perms) {
    if (FS.ignorePermissions) {
      return 0;
    }
    if (perms.includes("r") && !(node.mode & 292)) {
      return 2;
    }
    if (perms.includes("w") && !(node.mode & 146)) {
      return 2;
    }
    if (perms.includes("x") && !(node.mode & 73)) {
      return 2;
    }
    return 0;
  }, mayLookup(dir) {
    if (!FS.isDir(dir.mode))
      return 54;
    var errCode = FS.nodePermissions(dir, "x");
    if (errCode)
      return errCode;
    if (!dir.node_ops.lookup)
      return 2;
    return 0;
  }, mayCreate(dir, name) {
    if (!FS.isDir(dir.mode)) {
      return 54;
    }
    try {
      var node = FS.lookupNode(dir, name);
      return 20;
    } catch (e) {}
    return FS.nodePermissions(dir, "wx");
  }, mayDelete(dir, name, isdir) {
    var node;
    try {
      node = FS.lookupNode(dir, name);
    } catch (e) {
      return e.errno;
    }
    var errCode = FS.nodePermissions(dir, "wx");
    if (errCode) {
      return errCode;
    }
    if (isdir) {
      if (!FS.isDir(node.mode)) {
        return 54;
      }
      if (FS.isRoot(node) || FS.getPath(node) === FS.cwd()) {
        return 10;
      }
    } else if (FS.isDir(node.mode)) {
      return 31;
    }
    return 0;
  }, mayOpen(node, flags) {
    if (!node) {
      return 44;
    }
    if (FS.isLink(node.mode)) {
      return 32;
    }
    var mode = FS.flagsToPermissionString(flags);
    if (FS.isDir(node.mode)) {
      if (mode !== "r" || flags & (512 | 64)) {
        return 31;
      }
    }
    return FS.nodePermissions(node, mode);
  }, checkOpExists(op, err2) {
    if (!op) {
      throw new FS.ErrnoError(err2);
    }
    return op;
  }, MAX_OPEN_FDS: 4096, nextfd() {
    for (var fd = 0;fd <= FS.MAX_OPEN_FDS; fd++) {
      if (!FS.streams[fd]) {
        return fd;
      }
    }
    throw new FS.ErrnoError(33);
  }, getStreamChecked(fd) {
    var stream = FS.getStream(fd);
    if (!stream) {
      throw new FS.ErrnoError(8);
    }
    return stream;
  }, getStream: (fd) => FS.streams[fd], createStream(stream, fd = -1) {
    stream = Object.assign(new FS.FSStream, stream);
    if (fd == -1) {
      fd = FS.nextfd();
    }
    stream.fd = fd;
    FS.streams[fd] = stream;
    return stream;
  }, closeStream(fd) {
    FS.streams[fd] = null;
  }, dupStream(origStream, fd = -1) {
    var stream = FS.createStream(origStream, fd);
    stream.stream_ops?.dup?.(stream);
    return stream;
  }, doSetAttr(stream, node, attr) {
    var setattr = stream?.stream_ops.setattr;
    var arg = setattr ? stream : node;
    setattr ??= node.node_ops.setattr;
    FS.checkOpExists(setattr, 63);
    try {
      setattr(arg, attr);
    } catch (e) {
      if (e instanceof RangeError) {
        throw new FS.ErrnoError(22);
      }
      throw e;
    }
  }, chrdev_stream_ops: { open(stream) {
    var device = FS.getDevice(stream.node.rdev);
    stream.stream_ops = device.stream_ops;
    stream.stream_ops.open?.(stream);
  }, llseek() {
    throw new FS.ErrnoError(70);
  } }, major: (dev) => dev >> 8, minor: (dev) => dev & 255, makedev: (ma, mi) => ma << 8 | mi, registerDevice(dev, ops) {
    FS.devices[dev] = { stream_ops: ops };
  }, getDevice: (dev) => FS.devices[dev], getMounts(mount) {
    var mounts = [];
    var check = [mount];
    while (check.length) {
      var m = check.pop();
      mounts.push(m);
      check.push(...m.mounts);
    }
    return mounts;
  }, syncfs(populate, callback) {
    if (typeof populate == "function") {
      callback = populate;
      populate = false;
    }
    FS.syncFSRequests++;
    if (FS.syncFSRequests > 1) {
      err(`warning: ${FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`);
    }
    var mounts = FS.getMounts(FS.root.mount);
    var completed = 0;
    function doCallback(errCode) {
      FS.syncFSRequests--;
      return callback(errCode);
    }
    function done(errCode) {
      if (errCode) {
        if (!done.errored) {
          done.errored = true;
          return doCallback(errCode);
        }
        return;
      }
      if (++completed >= mounts.length) {
        doCallback(null);
      }
    }
    for (var mount of mounts) {
      if (mount.type.syncfs) {
        mount.type.syncfs(mount, populate, done);
      } else {
        done(null);
      }
    }
  }, mount(type, opts, mountpoint) {
    var root = mountpoint === "/";
    var pseudo = !mountpoint;
    var node;
    if (root && FS.root) {
      throw new FS.ErrnoError(10);
    } else if (!root && !pseudo) {
      var lookup = FS.lookupPath(mountpoint, { follow_mount: false });
      mountpoint = lookup.path;
      node = lookup.node;
      if (FS.isMountpoint(node)) {
        throw new FS.ErrnoError(10);
      }
      if (!FS.isDir(node.mode)) {
        throw new FS.ErrnoError(54);
      }
    }
    var mount = { type, opts, mountpoint, mounts: [] };
    var mountRoot = type.mount(mount);
    mountRoot.mount = mount;
    mount.root = mountRoot;
    if (root) {
      FS.root = mountRoot;
    } else if (node) {
      node.mounted = mount;
      if (node.mount) {
        node.mount.mounts.push(mount);
      }
    }
    return mountRoot;
  }, unmount(mountpoint) {
    var lookup = FS.lookupPath(mountpoint, { follow_mount: false });
    if (!FS.isMountpoint(lookup.node)) {
      throw new FS.ErrnoError(28);
    }
    var node = lookup.node;
    var mount = node.mounted;
    var mounts = FS.getMounts(mount);
    for (var [hash, current] of Object.entries(FS.nameTable)) {
      while (current) {
        var next = current.name_next;
        if (mounts.includes(current.mount)) {
          FS.destroyNode(current);
        }
        current = next;
      }
    }
    node.mounted = null;
    var idx = node.mount.mounts.indexOf(mount);
    node.mount.mounts.splice(idx, 1);
  }, lookup(parent, name) {
    return parent.node_ops.lookup(parent, name);
  }, mknod(path, mode, dev) {
    var lookup = FS.lookupPath(path, { parent: true });
    var parent = lookup.node;
    var name = PATH.basename(path);
    if (!name) {
      throw new FS.ErrnoError(28);
    }
    if (name === "." || name === "..") {
      throw new FS.ErrnoError(20);
    }
    var errCode = FS.mayCreate(parent, name);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    if (!parent.node_ops.mknod) {
      throw new FS.ErrnoError(63);
    }
    return parent.node_ops.mknod(parent, name, mode, dev);
  }, statfs(path) {
    return FS.statfsNode(FS.lookupPath(path, { follow: true }).node);
  }, statfsStream(stream) {
    return FS.statfsNode(stream.node);
  }, statfsNode(node) {
    var rtn = { bsize: 4096, frsize: 4096, blocks: 1e6, bfree: 500000, bavail: 500000, files: FS.nextInode, ffree: FS.nextInode - 1, fsid: 42, flags: 2, namelen: 255 };
    if (node.node_ops.statfs) {
      Object.assign(rtn, node.node_ops.statfs(node.mount.opts.root));
    }
    return rtn;
  }, create(path, mode = 438) {
    mode &= 4095;
    mode |= 32768;
    return FS.mknod(path, mode, 0);
  }, mkdir(path, mode = 511) {
    mode &= 511 | 512;
    mode |= 16384;
    return FS.mknod(path, mode, 0);
  }, mkdirTree(path, mode) {
    var dirs = path.split("/");
    var d = "";
    for (var dir of dirs) {
      if (!dir)
        continue;
      if (d || PATH.isAbs(path))
        d += "/";
      d += dir;
      try {
        FS.mkdir(d, mode);
      } catch (e) {
        if (e.errno != 20)
          throw e;
      }
    }
  }, mkdev(path, mode, dev) {
    if (typeof dev == "undefined") {
      dev = mode;
      mode = 438;
    }
    mode |= 8192;
    return FS.mknod(path, mode, dev);
  }, symlink(oldpath, newpath) {
    if (!PATH_FS.resolve(oldpath)) {
      throw new FS.ErrnoError(44);
    }
    var lookup = FS.lookupPath(newpath, { parent: true });
    var parent = lookup.node;
    if (!parent) {
      throw new FS.ErrnoError(44);
    }
    var newname = PATH.basename(newpath);
    var errCode = FS.mayCreate(parent, newname);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    if (!parent.node_ops.symlink) {
      throw new FS.ErrnoError(63);
    }
    return parent.node_ops.symlink(parent, newname, oldpath);
  }, rename(old_path, new_path) {
    var old_dirname = PATH.dirname(old_path);
    var new_dirname = PATH.dirname(new_path);
    var old_name = PATH.basename(old_path);
    var new_name = PATH.basename(new_path);
    var lookup, old_dir, new_dir;
    lookup = FS.lookupPath(old_path, { parent: true });
    old_dir = lookup.node;
    lookup = FS.lookupPath(new_path, { parent: true });
    new_dir = lookup.node;
    if (!old_dir || !new_dir)
      throw new FS.ErrnoError(44);
    if (old_dir.mount !== new_dir.mount) {
      throw new FS.ErrnoError(75);
    }
    var old_node = FS.lookupNode(old_dir, old_name);
    var relative = PATH_FS.relative(old_path, new_dirname);
    if (relative.charAt(0) !== ".") {
      throw new FS.ErrnoError(28);
    }
    relative = PATH_FS.relative(new_path, old_dirname);
    if (relative.charAt(0) !== ".") {
      throw new FS.ErrnoError(55);
    }
    var new_node;
    try {
      new_node = FS.lookupNode(new_dir, new_name);
    } catch (e) {}
    if (old_node === new_node) {
      return;
    }
    var isdir = FS.isDir(old_node.mode);
    var errCode = FS.mayDelete(old_dir, old_name, isdir);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    errCode = new_node ? FS.mayDelete(new_dir, new_name, isdir) : FS.mayCreate(new_dir, new_name);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    if (!old_dir.node_ops.rename) {
      throw new FS.ErrnoError(63);
    }
    if (FS.isMountpoint(old_node) || new_node && FS.isMountpoint(new_node)) {
      throw new FS.ErrnoError(10);
    }
    if (new_dir !== old_dir) {
      errCode = FS.nodePermissions(old_dir, "w");
      if (errCode) {
        throw new FS.ErrnoError(errCode);
      }
    }
    FS.hashRemoveNode(old_node);
    try {
      old_dir.node_ops.rename(old_node, new_dir, new_name);
      old_node.parent = new_dir;
    } catch (e) {
      throw e;
    } finally {
      FS.hashAddNode(old_node);
    }
  }, rmdir(path) {
    var lookup = FS.lookupPath(path, { parent: true });
    var parent = lookup.node;
    var name = PATH.basename(path);
    var node = FS.lookupNode(parent, name);
    var errCode = FS.mayDelete(parent, name, true);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    if (!parent.node_ops.rmdir) {
      throw new FS.ErrnoError(63);
    }
    if (FS.isMountpoint(node)) {
      throw new FS.ErrnoError(10);
    }
    parent.node_ops.rmdir(parent, name);
    FS.destroyNode(node);
  }, readdir(path) {
    var lookup = FS.lookupPath(path, { follow: true });
    var node = lookup.node;
    var readdir = FS.checkOpExists(node.node_ops.readdir, 54);
    return readdir(node);
  }, unlink(path) {
    var lookup = FS.lookupPath(path, { parent: true });
    var parent = lookup.node;
    if (!parent) {
      throw new FS.ErrnoError(44);
    }
    var name = PATH.basename(path);
    var node = FS.lookupNode(parent, name);
    var errCode = FS.mayDelete(parent, name, false);
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    if (!parent.node_ops.unlink) {
      throw new FS.ErrnoError(63);
    }
    if (FS.isMountpoint(node)) {
      throw new FS.ErrnoError(10);
    }
    parent.node_ops.unlink(parent, name);
    FS.destroyNode(node);
  }, readlink(path) {
    var lookup = FS.lookupPath(path);
    var link = lookup.node;
    if (!link) {
      throw new FS.ErrnoError(44);
    }
    if (!link.node_ops.readlink) {
      throw new FS.ErrnoError(28);
    }
    return link.node_ops.readlink(link);
  }, stat(path, dontFollow) {
    var lookup = FS.lookupPath(path, { follow: !dontFollow });
    var node = lookup.node;
    var getattr = FS.checkOpExists(node.node_ops.getattr, 63);
    return getattr(node);
  }, fstat(fd) {
    var stream = FS.getStreamChecked(fd);
    var node = stream.node;
    var getattr = stream.stream_ops.getattr;
    var arg = getattr ? stream : node;
    getattr ??= node.node_ops.getattr;
    FS.checkOpExists(getattr, 63);
    return getattr(arg);
  }, lstat(path) {
    return FS.stat(path, true);
  }, doChmod(stream, node, mode, dontFollow) {
    FS.doSetAttr(stream, node, { mode: mode & 4095 | node.mode & ~4095, ctime: Date.now(), dontFollow });
  }, chmod(path, mode, dontFollow) {
    var node;
    if (typeof path == "string") {
      var lookup = FS.lookupPath(path, { follow: !dontFollow });
      node = lookup.node;
    } else {
      node = path;
    }
    FS.doChmod(null, node, mode, dontFollow);
  }, lchmod(path, mode) {
    FS.chmod(path, mode, true);
  }, fchmod(fd, mode) {
    var stream = FS.getStreamChecked(fd);
    FS.doChmod(stream, stream.node, mode, false);
  }, doChown(stream, node, dontFollow) {
    FS.doSetAttr(stream, node, { timestamp: Date.now(), dontFollow });
  }, chown(path, uid, gid, dontFollow) {
    var node;
    if (typeof path == "string") {
      var lookup = FS.lookupPath(path, { follow: !dontFollow });
      node = lookup.node;
    } else {
      node = path;
    }
    FS.doChown(null, node, dontFollow);
  }, lchown(path, uid, gid) {
    FS.chown(path, uid, gid, true);
  }, fchown(fd, uid, gid) {
    var stream = FS.getStreamChecked(fd);
    FS.doChown(stream, stream.node, false);
  }, doTruncate(stream, node, len) {
    if (FS.isDir(node.mode)) {
      throw new FS.ErrnoError(31);
    }
    if (!FS.isFile(node.mode)) {
      throw new FS.ErrnoError(28);
    }
    var errCode = FS.nodePermissions(node, "w");
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    FS.doSetAttr(stream, node, { size: len, timestamp: Date.now() });
  }, truncate(path, len) {
    if (len < 0) {
      throw new FS.ErrnoError(28);
    }
    var node;
    if (typeof path == "string") {
      var lookup = FS.lookupPath(path, { follow: true });
      node = lookup.node;
    } else {
      node = path;
    }
    FS.doTruncate(null, node, len);
  }, ftruncate(fd, len) {
    var stream = FS.getStreamChecked(fd);
    if (len < 0 || (stream.flags & 2097155) === 0) {
      throw new FS.ErrnoError(28);
    }
    FS.doTruncate(stream, stream.node, len);
  }, utime(path, atime, mtime) {
    var lookup = FS.lookupPath(path, { follow: true });
    var node = lookup.node;
    var setattr = FS.checkOpExists(node.node_ops.setattr, 63);
    setattr(node, { atime, mtime });
  }, open(path, flags, mode = 438) {
    if (path === "") {
      throw new FS.ErrnoError(44);
    }
    flags = FS_modeStringToFlags(flags);
    if (flags & 64) {
      mode = mode & 4095 | 32768;
    } else {
      mode = 0;
    }
    var node;
    var isDirPath;
    if (typeof path == "object") {
      node = path;
    } else {
      isDirPath = path.endsWith("/");
      var lookup = FS.lookupPath(path, { follow: !(flags & 131072), noent_okay: true });
      node = lookup.node;
      path = lookup.path;
    }
    var created = false;
    if (flags & 64) {
      if (node) {
        if (flags & 128) {
          throw new FS.ErrnoError(20);
        }
      } else if (isDirPath) {
        throw new FS.ErrnoError(31);
      } else {
        node = FS.mknod(path, mode | 511, 0);
        created = true;
      }
    }
    if (!node) {
      throw new FS.ErrnoError(44);
    }
    if (FS.isChrdev(node.mode)) {
      flags &= ~512;
    }
    if (flags & 65536 && !FS.isDir(node.mode)) {
      throw new FS.ErrnoError(54);
    }
    if (!created) {
      var errCode = FS.mayOpen(node, flags);
      if (errCode) {
        throw new FS.ErrnoError(errCode);
      }
    }
    if (flags & 512 && !created) {
      FS.truncate(node, 0);
    }
    flags &= ~(128 | 512 | 131072);
    var stream = FS.createStream({ node, path: FS.getPath(node), flags, seekable: true, position: 0, stream_ops: node.stream_ops, ungotten: [], error: false });
    if (stream.stream_ops.open) {
      stream.stream_ops.open(stream);
    }
    if (created) {
      FS.chmod(node, mode & 511);
    }
    return stream;
  }, close(stream) {
    if (FS.isClosed(stream)) {
      throw new FS.ErrnoError(8);
    }
    if (stream.getdents)
      stream.getdents = null;
    try {
      if (stream.stream_ops.close) {
        stream.stream_ops.close(stream);
      }
    } catch (e) {
      throw e;
    } finally {
      FS.closeStream(stream.fd);
    }
    stream.fd = null;
  }, isClosed(stream) {
    return stream.fd === null;
  }, llseek(stream, offset, whence) {
    if (FS.isClosed(stream)) {
      throw new FS.ErrnoError(8);
    }
    if (!stream.seekable || !stream.stream_ops.llseek) {
      throw new FS.ErrnoError(70);
    }
    if (whence != 0 && whence != 1 && whence != 2) {
      throw new FS.ErrnoError(28);
    }
    stream.position = stream.stream_ops.llseek(stream, offset, whence);
    stream.ungotten = [];
    return stream.position;
  }, read(stream, buffer, offset, length, position) {
    if (length < 0 || position < 0) {
      throw new FS.ErrnoError(28);
    }
    if (FS.isClosed(stream)) {
      throw new FS.ErrnoError(8);
    }
    if ((stream.flags & 2097155) === 1) {
      throw new FS.ErrnoError(8);
    }
    if (FS.isDir(stream.node.mode)) {
      throw new FS.ErrnoError(31);
    }
    if (!stream.stream_ops.read) {
      throw new FS.ErrnoError(28);
    }
    var seeking = typeof position != "undefined";
    if (!seeking) {
      position = stream.position;
    } else if (!stream.seekable) {
      throw new FS.ErrnoError(70);
    }
    var bytesRead = stream.stream_ops.read(stream, buffer, offset, length, position);
    if (!seeking)
      stream.position += bytesRead;
    return bytesRead;
  }, write(stream, buffer, offset, length, position, canOwn) {
    if (length < 0 || position < 0) {
      throw new FS.ErrnoError(28);
    }
    if (FS.isClosed(stream)) {
      throw new FS.ErrnoError(8);
    }
    if ((stream.flags & 2097155) === 0) {
      throw new FS.ErrnoError(8);
    }
    if (FS.isDir(stream.node.mode)) {
      throw new FS.ErrnoError(31);
    }
    if (!stream.stream_ops.write) {
      throw new FS.ErrnoError(28);
    }
    if (stream.seekable && stream.flags & 1024) {
      FS.llseek(stream, 0, 2);
    }
    var seeking = typeof position != "undefined";
    if (!seeking) {
      position = stream.position;
    } else if (!stream.seekable) {
      throw new FS.ErrnoError(70);
    }
    var bytesWritten = stream.stream_ops.write(stream, buffer, offset, length, position, canOwn);
    if (!seeking)
      stream.position += bytesWritten;
    return bytesWritten;
  }, mmap(stream, length, position, prot, flags) {
    if ((prot & 2) !== 0 && (flags & 2) === 0 && (stream.flags & 2097155) !== 2) {
      throw new FS.ErrnoError(2);
    }
    if ((stream.flags & 2097155) === 1) {
      throw new FS.ErrnoError(2);
    }
    if (!stream.stream_ops.mmap) {
      throw new FS.ErrnoError(43);
    }
    if (!length) {
      throw new FS.ErrnoError(28);
    }
    return stream.stream_ops.mmap(stream, length, position, prot, flags);
  }, msync(stream, buffer, offset, length, mmapFlags) {
    if (!stream.stream_ops.msync) {
      return 0;
    }
    return stream.stream_ops.msync(stream, buffer, offset, length, mmapFlags);
  }, ioctl(stream, cmd, arg) {
    if (!stream.stream_ops.ioctl) {
      throw new FS.ErrnoError(59);
    }
    return stream.stream_ops.ioctl(stream, cmd, arg);
  }, readFile(path, opts = {}) {
    opts.flags = opts.flags || 0;
    opts.encoding = opts.encoding || "binary";
    if (opts.encoding !== "utf8" && opts.encoding !== "binary") {
      abort(`Invalid encoding type "${opts.encoding}"`);
    }
    var stream = FS.open(path, opts.flags);
    var stat = FS.stat(path);
    var length = stat.size;
    var buf = new Uint8Array(length);
    FS.read(stream, buf, 0, length, 0);
    if (opts.encoding === "utf8") {
      buf = UTF8ArrayToString(buf);
    }
    FS.close(stream);
    return buf;
  }, writeFile(path, data, opts = {}) {
    opts.flags = opts.flags || 577;
    var stream = FS.open(path, opts.flags, opts.mode);
    data = FS_fileDataToTypedArray(data);
    FS.write(stream, data, 0, data.byteLength, undefined, opts.canOwn);
    FS.close(stream);
  }, cwd: () => FS.currentPath, chdir(path) {
    var lookup = FS.lookupPath(path, { follow: true });
    if (lookup.node === null) {
      throw new FS.ErrnoError(44);
    }
    if (!FS.isDir(lookup.node.mode)) {
      throw new FS.ErrnoError(54);
    }
    var errCode = FS.nodePermissions(lookup.node, "x");
    if (errCode) {
      throw new FS.ErrnoError(errCode);
    }
    FS.currentPath = lookup.path;
  }, createDefaultDirectories() {
    FS.mkdir("/tmp");
    FS.mkdir("/home");
    FS.mkdir("/home/web_user");
  }, createDefaultDevices() {
    FS.mkdir("/dev");
    FS.registerDevice(FS.makedev(1, 3), { read: () => 0, write: (stream, buffer, offset, length, pos) => length, llseek: () => 0 });
    FS.mkdev("/dev/null", FS.makedev(1, 3));
    TTY.register(FS.makedev(5, 0), TTY.default_tty_ops);
    TTY.register(FS.makedev(6, 0), TTY.default_tty1_ops);
    FS.mkdev("/dev/tty", FS.makedev(5, 0));
    FS.mkdev("/dev/tty1", FS.makedev(6, 0));
    var randomBuffer = new Uint8Array(1024), randomLeft = 0;
    var randomByte = () => {
      if (randomLeft === 0) {
        randomFill(randomBuffer);
        randomLeft = randomBuffer.byteLength;
      }
      return randomBuffer[--randomLeft];
    };
    FS.createDevice("/dev", "random", randomByte);
    FS.createDevice("/dev", "urandom", randomByte);
    FS.mkdir("/dev/shm");
    FS.mkdir("/dev/shm/tmp");
  }, createSpecialDirectories() {
    FS.mkdir("/proc");
    var proc_self = FS.mkdir("/proc/self");
    FS.mkdir("/proc/self/fd");
    FS.mount({ mount() {
      var node = FS.createNode(proc_self, "fd", 16895, 73);
      node.stream_ops = { llseek: MEMFS.stream_ops.llseek };
      node.node_ops = { lookup(parent, name) {
        var fd = +name;
        var stream = FS.getStreamChecked(fd);
        var ret = { parent: null, mount: { mountpoint: "fake" }, node_ops: { readlink: () => stream.path }, id: fd + 1 };
        ret.parent = ret;
        return ret;
      }, readdir() {
        return Array.from(FS.streams.entries()).filter(([k, v]) => v).map(([k, v]) => k.toString());
      } };
      return node;
    } }, {}, "/proc/self/fd");
  }, createStandardStreams(input, output, error) {
    if (input) {
      FS.createDevice("/dev", "stdin", input);
    } else {
      FS.symlink("/dev/tty", "/dev/stdin");
    }
    if (output) {
      FS.createDevice("/dev", "stdout", null, output);
    } else {
      FS.symlink("/dev/tty", "/dev/stdout");
    }
    if (error) {
      FS.createDevice("/dev", "stderr", null, error);
    } else {
      FS.symlink("/dev/tty1", "/dev/stderr");
    }
    var stdin = FS.open("/dev/stdin", 0);
    var stdout = FS.open("/dev/stdout", 1);
    var stderr = FS.open("/dev/stderr", 1);
  }, staticInit() {
    FS.nameTable = new Array(4096);
    FS.mount(MEMFS, {}, "/");
    FS.createDefaultDirectories();
    FS.createDefaultDevices();
    FS.createSpecialDirectories();
    FS.filesystems = { MEMFS };
  }, init(input, output, error) {
    FS.initialized = true;
    input ??= Module2["stdin"];
    output ??= Module2["stdout"];
    error ??= Module2["stderr"];
    FS.createStandardStreams(input, output, error);
  }, quit() {
    FS.initialized = false;
    for (var stream of FS.streams) {
      if (stream) {
        FS.close(stream);
      }
    }
  }, findObject(path, dontResolveLastLink) {
    var ret = FS.analyzePath(path, dontResolveLastLink);
    if (!ret.exists) {
      return null;
    }
    return ret.object;
  }, analyzePath(path, dontResolveLastLink) {
    try {
      var lookup = FS.lookupPath(path, { follow: !dontResolveLastLink });
      path = lookup.path;
    } catch (e) {}
    var ret = { isRoot: false, exists: false, error: 0, name: null, path: null, object: null, parentExists: false, parentPath: null, parentObject: null };
    try {
      var lookup = FS.lookupPath(path, { parent: true });
      ret.parentExists = true;
      ret.parentPath = lookup.path;
      ret.parentObject = lookup.node;
      ret.name = PATH.basename(path);
      lookup = FS.lookupPath(path, { follow: !dontResolveLastLink });
      ret.exists = true;
      ret.path = lookup.path;
      ret.object = lookup.node;
      ret.name = lookup.node.name;
      ret.isRoot = lookup.path === "/";
    } catch (e) {
      ret.error = e.errno;
    }
    return ret;
  }, createPath(parent, path, canRead, canWrite) {
    parent = typeof parent == "string" ? parent : FS.getPath(parent);
    var parts = path.split("/").reverse();
    while (parts.length) {
      var part = parts.pop();
      if (!part)
        continue;
      var current = PATH.join2(parent, part);
      try {
        FS.mkdir(current);
      } catch (e) {
        if (e.errno != 20)
          throw e;
      }
      parent = current;
    }
    return current;
  }, createFile(parent, name, properties, canRead, canWrite) {
    var path = PATH.join2(typeof parent == "string" ? parent : FS.getPath(parent), name);
    var mode = FS_getMode(canRead, canWrite);
    return FS.create(path, mode);
  }, createDataFile(parent, name, data, canRead, canWrite, canOwn) {
    var path = name;
    if (parent) {
      parent = typeof parent == "string" ? parent : FS.getPath(parent);
      path = name ? PATH.join2(parent, name) : parent;
    }
    var mode = FS_getMode(canRead, canWrite);
    var node = FS.create(path, mode);
    if (data) {
      data = FS_fileDataToTypedArray(data);
      FS.chmod(node, mode | 146);
      var stream = FS.open(node, 577);
      FS.write(stream, data, 0, data.length, 0, canOwn);
      FS.close(stream);
      FS.chmod(node, mode);
    }
  }, createDevice(parent, name, input, output) {
    var path = PATH.join2(typeof parent == "string" ? parent : FS.getPath(parent), name);
    var mode = FS_getMode(!!input, !!output);
    FS.createDevice.major ??= 64;
    var dev = FS.makedev(FS.createDevice.major++, 0);
    FS.registerDevice(dev, { open(stream) {
      stream.seekable = false;
    }, close(stream) {
      if (output?.buffer?.length) {
        output(10);
      }
    }, read(stream, buffer, offset, length, pos) {
      var bytesRead = 0;
      for (var i = 0;i < length; i++) {
        var result;
        try {
          result = input();
        } catch (e) {
          throw new FS.ErrnoError(29);
        }
        if (result === undefined && bytesRead === 0) {
          throw new FS.ErrnoError(6);
        }
        if (result === null || result === undefined)
          break;
        bytesRead++;
        buffer[offset + i] = result;
      }
      if (bytesRead) {
        stream.node.atime = Date.now();
      }
      return bytesRead;
    }, write(stream, buffer, offset, length, pos) {
      for (var i = 0;i < length; i++) {
        try {
          output(buffer[offset + i]);
        } catch (e) {
          throw new FS.ErrnoError(29);
        }
      }
      if (length) {
        stream.node.mtime = stream.node.ctime = Date.now();
      }
      return i;
    } });
    return FS.mkdev(path, mode, dev);
  }, forceLoadFile(obj) {
    if (obj.isDevice || obj.isFolder || obj.link || obj.contents)
      return true;
    if (globalThis.XMLHttpRequest) {
      abort("Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.");
    } else {
      try {
        obj.contents = readBinary(obj.url);
      } catch (e) {
        throw new FS.ErrnoError(29);
      }
    }
  }, createLazyFile(parent, name, url, canRead, canWrite) {

    class LazyUint8Array {
      lengthKnown = false;
      chunks = [];
      get(idx) {
        if (idx > this.length - 1 || idx < 0) {
          return;
        }
        var chunkOffset = idx % this.chunkSize;
        var chunkNum = idx / this.chunkSize | 0;
        return this.getter(chunkNum)[chunkOffset];
      }
      setDataGetter(getter) {
        this.getter = getter;
      }
      cacheLength() {
        var xhr = new XMLHttpRequest;
        xhr.open("HEAD", url, false);
        xhr.send(null);
        if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304))
          abort("Couldn't load " + url + ". Status: " + xhr.status);
        var datalength = Number(xhr.getResponseHeader("Content-length"));
        var header;
        var hasByteServing = (header = xhr.getResponseHeader("Accept-Ranges")) && header === "bytes";
        var usesGzip = (header = xhr.getResponseHeader("Content-Encoding")) && header === "gzip";
        var chunkSize = 1024 * 1024;
        if (!hasByteServing)
          chunkSize = datalength;
        var doXHR = (from, to) => {
          if (from > to)
            abort("invalid range (" + from + ", " + to + ") or no bytes requested!");
          if (to > datalength - 1)
            abort("only " + datalength + " bytes available! programmer error!");
          var xhr2 = new XMLHttpRequest;
          xhr2.open("GET", url, false);
          if (datalength !== chunkSize)
            xhr2.setRequestHeader("Range", "bytes=" + from + "-" + to);
          xhr2.responseType = "arraybuffer";
          if (xhr2.overrideMimeType) {
            xhr2.overrideMimeType("text/plain; charset=x-user-defined");
          }
          xhr2.send(null);
          if (!(xhr2.status >= 200 && xhr2.status < 300 || xhr2.status === 304))
            abort("Couldn't load " + url + ". Status: " + xhr2.status);
          if (xhr2.response !== undefined) {
            return new Uint8Array(xhr2.response || []);
          }
          return intArrayFromString(xhr2.responseText || "", true);
        };
        var lazyArray2 = this;
        lazyArray2.setDataGetter((chunkNum) => {
          var start = chunkNum * chunkSize;
          var end = (chunkNum + 1) * chunkSize - 1;
          end = Math.min(end, datalength - 1);
          if (typeof lazyArray2.chunks[chunkNum] == "undefined") {
            lazyArray2.chunks[chunkNum] = doXHR(start, end);
          }
          if (typeof lazyArray2.chunks[chunkNum] == "undefined")
            abort("doXHR failed!");
          return lazyArray2.chunks[chunkNum];
        });
        if (usesGzip || !datalength) {
          chunkSize = datalength = 1;
          datalength = this.getter(0).length;
          chunkSize = datalength;
          out("LazyFiles on gzip forces download of the whole file when length is accessed");
        }
        this._length = datalength;
        this._chunkSize = chunkSize;
        this.lengthKnown = true;
      }
      get length() {
        if (!this.lengthKnown) {
          this.cacheLength();
        }
        return this._length;
      }
      get chunkSize() {
        if (!this.lengthKnown) {
          this.cacheLength();
        }
        return this._chunkSize;
      }
    }
    if (globalThis.XMLHttpRequest) {
      if (!ENVIRONMENT_IS_WORKER)
        abort("Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc");
      var lazyArray = new LazyUint8Array;
      var properties = { isDevice: false, contents: lazyArray };
    } else {
      var properties = { isDevice: false, url };
    }
    var node = FS.createFile(parent, name, properties, canRead, canWrite);
    if (properties.contents) {
      node.contents = properties.contents;
    } else if (properties.url) {
      node.contents = null;
      node.url = properties.url;
    }
    Object.defineProperties(node, { usedBytes: { get: function() {
      return this.contents.length;
    } } });
    var stream_ops = {};
    for (const [key, fn] of Object.entries(node.stream_ops)) {
      stream_ops[key] = (...args) => {
        FS.forceLoadFile(node);
        return fn(...args);
      };
    }
    function writeChunks(stream, buffer, offset, length, position) {
      var contents = stream.node.contents;
      if (position >= contents.length)
        return 0;
      var size = Math.min(contents.length - position, length);
      if (contents.slice) {
        for (var i = 0;i < size; i++) {
          buffer[offset + i] = contents[position + i];
        }
      } else {
        for (var i = 0;i < size; i++) {
          buffer[offset + i] = contents.get(position + i);
        }
      }
      return size;
    }
    stream_ops.read = (stream, buffer, offset, length, position) => {
      FS.forceLoadFile(node);
      return writeChunks(stream, buffer, offset, length, position);
    };
    stream_ops.mmap = (stream, length, position, prot, flags) => {
      FS.forceLoadFile(node);
      var ptr = mmapAlloc(length);
      if (!ptr) {
        throw new FS.ErrnoError(48);
      }
      writeChunks(stream, HEAP8, ptr, length, position);
      return { ptr, allocated: true };
    };
    node.stream_ops = stream_ops;
    return node;
  } };
  var UTF8ToString = (ptr, maxBytesToRead, ignoreNul) => {
    ptr >>>= 0;
    return ptr ? UTF8ArrayToString(HEAPU8, ptr, maxBytesToRead, ignoreNul) : "";
  };
  var SYSCALLS = { currentUmask: 18, calculateAt(dirfd, path, allowEmpty) {
    if (PATH.isAbs(path)) {
      return path;
    }
    var dir;
    if (dirfd === -100) {
      dir = FS.cwd();
    } else {
      var dirstream = SYSCALLS.getStreamFromFD(dirfd);
      dir = dirstream.path;
    }
    if (path.length == 0) {
      if (!allowEmpty) {
        throw new FS.ErrnoError(44);
      }
      return dir;
    }
    return dir + "/" + path;
  }, writeStat(buf, stat) {
    HEAPU32[buf >>> 2 >>> 0] = stat.dev;
    HEAPU32[buf + 4 >>> 2 >>> 0] = stat.mode;
    HEAPU32[buf + 8 >>> 2 >>> 0] = stat.nlink;
    HEAPU32[buf + 12 >>> 2 >>> 0] = stat.uid;
    HEAPU32[buf + 16 >>> 2 >>> 0] = stat.gid;
    HEAPU32[buf + 20 >>> 2 >>> 0] = stat.rdev;
    HEAP64[buf + 24 >>> 3 >>> 0] = BigInt(stat.size);
    HEAP32[buf + 32 >>> 2 >>> 0] = 4096;
    HEAP32[buf + 36 >>> 2 >>> 0] = stat.blocks;
    var atime = stat.atime.getTime();
    var mtime = stat.mtime.getTime();
    var ctime = stat.ctime.getTime();
    HEAP64[buf + 40 >>> 3 >>> 0] = BigInt(Math.floor(atime / 1000));
    HEAPU32[buf + 48 >>> 2 >>> 0] = atime % 1000 * 1000 * 1000;
    HEAP64[buf + 56 >>> 3 >>> 0] = BigInt(Math.floor(mtime / 1000));
    HEAPU32[buf + 64 >>> 2 >>> 0] = mtime % 1000 * 1000 * 1000;
    HEAP64[buf + 72 >>> 3 >>> 0] = BigInt(Math.floor(ctime / 1000));
    HEAPU32[buf + 80 >>> 2 >>> 0] = ctime % 1000 * 1000 * 1000;
    HEAP64[buf + 88 >>> 3 >>> 0] = BigInt(stat.ino);
    return 0;
  }, writeStatFs(buf, stats) {
    HEAPU32[buf + 4 >>> 2 >>> 0] = stats.bsize;
    HEAPU32[buf + 60 >>> 2 >>> 0] = stats.bsize;
    HEAP64[buf + 8 >>> 3 >>> 0] = BigInt(stats.blocks);
    HEAP64[buf + 16 >>> 3 >>> 0] = BigInt(stats.bfree);
    HEAP64[buf + 24 >>> 3 >>> 0] = BigInt(stats.bavail);
    HEAP64[buf + 32 >>> 3 >>> 0] = BigInt(stats.files);
    HEAP64[buf + 40 >>> 3 >>> 0] = BigInt(stats.ffree);
    HEAPU32[buf + 48 >>> 2 >>> 0] = stats.fsid;
    HEAPU32[buf + 64 >>> 2 >>> 0] = stats.flags;
    HEAPU32[buf + 56 >>> 2 >>> 0] = stats.namelen;
  }, doMsync(addr, stream, len, flags, offset) {
    if (!FS.isFile(stream.node.mode)) {
      throw new FS.ErrnoError(43);
    }
    if (flags & 2) {
      return 0;
    }
    var buffer = HEAPU8.slice(addr, addr + len);
    FS.msync(stream, buffer, offset, len, flags);
  }, getStreamFromFD(fd) {
    var stream = FS.getStreamChecked(fd);
    return stream;
  }, varargs: undefined, getStr(ptr) {
    var ret = UTF8ToString(ptr);
    return ret;
  } };
  var INT53_MAX = 9007199254740992;
  var INT53_MIN = -9007199254740992;
  var bigintToI53Checked = (num) => num < INT53_MIN || num > INT53_MAX ? NaN : Number(num);
  function ___syscall_fcntl64(fd, cmd, varargs) {
    varargs >>>= 0;
    SYSCALLS.varargs = varargs;
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      switch (cmd) {
        case 0: {
          var arg = syscallGetVarargI();
          if (arg < 0) {
            return -28;
          }
          while (FS.streams[arg]) {
            arg++;
          }
          var newStream;
          newStream = FS.dupStream(stream, arg);
          return newStream.fd;
        }
        case 1:
        case 2:
          return 0;
        case 3:
          return stream.flags;
        case 4: {
          var arg = syscallGetVarargI();
          var mask = 289792;
          stream.flags = stream.flags & ~mask | arg & mask;
          return 0;
        }
        case 12: {
          var arg = syscallGetVarargP();
          var offset = 0;
          HEAP16[arg + offset >>> 1 >>> 0] = 2;
          return 0;
        }
        case 13:
        case 14:
          return 0;
      }
      return -28;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  function ___syscall_ioctl(fd, op, varargs) {
    varargs >>>= 0;
    SYSCALLS.varargs = varargs;
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      switch (op) {
        case 21509: {
          if (!stream.tty)
            return -59;
          return 0;
        }
        case 21505: {
          if (!stream.tty)
            return -59;
          if (stream.tty.ops.ioctl_tcgets) {
            var termios = stream.tty.ops.ioctl_tcgets(stream);
            var argp = syscallGetVarargP();
            HEAP32[argp >>> 2 >>> 0] = termios.c_iflag || 0;
            HEAP32[argp + 4 >>> 2 >>> 0] = termios.c_oflag || 0;
            HEAP32[argp + 8 >>> 2 >>> 0] = termios.c_cflag || 0;
            HEAP32[argp + 12 >>> 2 >>> 0] = termios.c_lflag || 0;
            for (var i = 0;i < 32; i++) {
              HEAP8[argp + i + 17 >>> 0] = termios.c_cc[i] || 0;
            }
            return 0;
          }
          return 0;
        }
        case 21510:
        case 21511:
        case 21512: {
          if (!stream.tty)
            return -59;
          return 0;
        }
        case 21506:
        case 21507:
        case 21508: {
          if (!stream.tty)
            return -59;
          if (stream.tty.ops.ioctl_tcsets) {
            var argp = syscallGetVarargP();
            var c_iflag = HEAP32[argp >>> 2 >>> 0];
            var c_oflag = HEAP32[argp + 4 >>> 2 >>> 0];
            var c_cflag = HEAP32[argp + 8 >>> 2 >>> 0];
            var c_lflag = HEAP32[argp + 12 >>> 2 >>> 0];
            var c_cc = [];
            for (var i = 0;i < 32; i++) {
              c_cc.push(HEAP8[argp + i + 17 >>> 0]);
            }
            return stream.tty.ops.ioctl_tcsets(stream.tty, op, { c_iflag, c_oflag, c_cflag, c_lflag, c_cc });
          }
          return 0;
        }
        case 21519: {
          if (!stream.tty)
            return -59;
          var argp = syscallGetVarargP();
          HEAP32[argp >>> 2 >>> 0] = 0;
          return 0;
        }
        case 21520: {
          if (!stream.tty)
            return -59;
          return -28;
        }
        case 21537:
        case 21531: {
          var argp = syscallGetVarargP();
          return FS.ioctl(stream, op, argp);
        }
        case 21523: {
          if (!stream.tty)
            return -59;
          if (stream.tty.ops.ioctl_tiocgwinsz) {
            var winsize = stream.tty.ops.ioctl_tiocgwinsz(stream.tty);
            var argp = syscallGetVarargP();
            HEAP16[argp >>> 1 >>> 0] = winsize[0];
            HEAP16[argp + 2 >>> 1 >>> 0] = winsize[1];
          }
          return 0;
        }
        case 21524: {
          if (!stream.tty)
            return -59;
          return 0;
        }
        case 21515: {
          if (!stream.tty)
            return -59;
          return 0;
        }
        default:
          return -28;
      }
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  function ___syscall_openat(dirfd, path, flags, varargs) {
    path >>>= 0;
    varargs >>>= 0;
    SYSCALLS.varargs = varargs;
    try {
      path = SYSCALLS.getStr(path);
      path = SYSCALLS.calculateAt(dirfd, path);
      var mode = varargs ? syscallGetVarargI() : 0;
      if (flags & 64) {
        mode &= ~SYSCALLS.currentUmask;
      }
      return FS.open(path, flags, mode).fd;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  function ___syscall_rmdir(path) {
    path >>>= 0;
    try {
      path = SYSCALLS.getStr(path);
      FS.rmdir(path);
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  function ___syscall_unlinkat(dirfd, path, flags) {
    path >>>= 0;
    try {
      path = SYSCALLS.getStr(path);
      path = SYSCALLS.calculateAt(dirfd, path);
      if (!flags) {
        FS.unlink(path);
      } else if (flags === 512) {
        FS.rmdir(path);
      } else {
        return -28;
      }
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  var __abort_js = () => abort("");
  function __mmap_js(len, prot, flags, fd, offset, allocated, addr) {
    len >>>= 0;
    offset = bigintToI53Checked(offset);
    allocated >>>= 0;
    addr >>>= 0;
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      var res = FS.mmap(stream, len, offset, prot, flags);
      var ptr = res.ptr;
      HEAP32[allocated >>> 2 >>> 0] = res.allocated;
      HEAPU32[addr >>> 2 >>> 0] = ptr;
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  function __munmap_js(addr, len, prot, flags, fd, offset) {
    addr >>>= 0;
    len >>>= 0;
    offset = bigintToI53Checked(offset);
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      if (prot & 2) {
        SYSCALLS.doMsync(addr, stream, len, flags, offset);
      }
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return -e.errno;
    }
  }
  var stringToUTF8 = (str, outPtr, maxBytesToWrite) => stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);
  var __tzset_js = function(timezone, daylight, std_name, dst_name) {
    timezone >>>= 0;
    daylight >>>= 0;
    std_name >>>= 0;
    dst_name >>>= 0;
    var currentYear = new Date().getFullYear();
    var winter = new Date(currentYear, 0, 1);
    var summer = new Date(currentYear, 6, 1);
    var winterOffset = winter.getTimezoneOffset();
    var summerOffset = summer.getTimezoneOffset();
    var stdTimezoneOffset = Math.max(winterOffset, summerOffset);
    HEAPU32[timezone >>> 2 >>> 0] = stdTimezoneOffset * 60;
    HEAP32[daylight >>> 2 >>> 0] = Number(winterOffset != summerOffset);
    var extractZone = (timezoneOffset) => {
      var sign = timezoneOffset >= 0 ? "-" : "+";
      var absOffset = Math.abs(timezoneOffset);
      var hours = String(Math.floor(absOffset / 60)).padStart(2, "0");
      var minutes = String(absOffset % 60).padStart(2, "0");
      return `UTC${sign}${hours}${minutes}`;
    };
    var winterName = extractZone(winterOffset);
    var summerName = extractZone(summerOffset);
    if (summerOffset < winterOffset) {
      stringToUTF8(winterName, std_name, 17);
      stringToUTF8(summerName, dst_name, 17);
    } else {
      stringToUTF8(winterName, dst_name, 17);
      stringToUTF8(summerName, std_name, 17);
    }
  };
  var _emscripten_get_now = () => performance.now();
  var _emscripten_date_now = () => Date.now();
  var nowIsMonotonic = 1;
  var checkWasiClock = (clock_id) => clock_id >= 0 && clock_id <= 3;
  function _clock_time_get(clk_id, ignored_precision, ptime) {
    ignored_precision = bigintToI53Checked(ignored_precision);
    ptime >>>= 0;
    if (!checkWasiClock(clk_id)) {
      return 28;
    }
    var now;
    if (clk_id === 0) {
      now = _emscripten_date_now();
    } else if (nowIsMonotonic) {
      now = _emscripten_get_now();
    } else {
      return 52;
    }
    var nsec = Math.round(now * 1000 * 1000);
    HEAP64[ptime >>> 3 >>> 0] = BigInt(nsec);
    return 0;
  }
  var readEmAsmArgsArray = [];
  var readEmAsmArgs = (sigPtr, buf) => {
    readEmAsmArgsArray.length = 0;
    var ch;
    while (ch = HEAPU8[sigPtr++ >>> 0]) {
      var wide = ch != 105;
      wide &= ch != 112;
      buf += wide && buf % 8 ? 4 : 0;
      readEmAsmArgsArray.push(ch == 112 ? HEAPU32[buf >>> 2 >>> 0] : ch == 106 ? HEAP64[buf >>> 3 >>> 0] : ch == 105 ? HEAP32[buf >>> 2 >>> 0] : HEAPF64[buf >>> 3 >>> 0]);
      buf += wide ? 8 : 4;
    }
    return readEmAsmArgsArray;
  };
  var runEmAsmFunction = (code, sigPtr, argbuf) => {
    var args = readEmAsmArgs(sigPtr, argbuf);
    return ASM_CONSTS[code](...args);
  };
  function _emscripten_asm_const_int(code, sigPtr, argbuf) {
    code >>>= 0;
    sigPtr >>>= 0;
    argbuf >>>= 0;
    return runEmAsmFunction(code, sigPtr, argbuf);
  }
  var getHeapMax = () => 4294901760;
  function _emscripten_get_heap_max() {
    return getHeapMax();
  }
  var _emscripten_has_asyncify = () => 2;
  var growMemory = (size) => {
    var oldHeapSize = wasmMemory.buffer.byteLength;
    var pages = (size - oldHeapSize + 65535) / 65536 | 0;
    try {
      wasmMemory.grow(pages);
      updateMemoryViews();
      return 1;
    } catch (e) {}
  };
  function _emscripten_resize_heap(requestedSize) {
    requestedSize >>>= 0;
    var oldSize = HEAPU8.length;
    var maxHeapSize = getHeapMax();
    if (requestedSize > maxHeapSize) {
      return false;
    }
    for (var cutDown = 1;cutDown <= 4; cutDown *= 2) {
      var overGrownHeapSize = oldSize * (1 + 0.2 / cutDown);
      overGrownHeapSize = Math.min(overGrownHeapSize, requestedSize + 100663296);
      var newSize = Math.min(maxHeapSize, alignMemory(Math.max(requestedSize, overGrownHeapSize), 65536));
      var replacement = growMemory(newSize);
      if (replacement) {
        return true;
      }
    }
    return false;
  }
  var stackSave = () => _emscripten_stack_get_current();
  var stackRestore = (val) => __emscripten_stack_restore(val);
  var stackAlloc = (sz) => __emscripten_stack_alloc(sz);
  var stringToUTF8OnStack = (str) => {
    var size = lengthBytesUTF8(str) + 1;
    var ret = stackAlloc(size);
    stringToUTF8(str, ret, size);
    return ret;
  };
  var writeI53ToI64 = (ptr, num) => {
    HEAPU32[ptr >>> 2 >>> 0] = num;
    var lower = HEAPU32[ptr >>> 2 >>> 0];
    HEAPU32[ptr + 4 >>> 2 >>> 0] = (num - lower) / 4294967296;
  };
  var stringToNewUTF8 = (str) => {
    var size = lengthBytesUTF8(str) + 1;
    var ret = _malloc(size);
    if (ret)
      stringToUTF8(str, ret, size);
    return ret;
  };
  var readI53FromI64 = (ptr) => HEAPU32[ptr >>> 2 >>> 0] + HEAP32[ptr + 4 >>> 2 >>> 0] * 4294967296;
  var wasmTableMirror = [];
  var getWasmTableEntry = (funcPtr) => {
    var func = wasmTableMirror[funcPtr];
    if (!func) {
      wasmTableMirror[funcPtr] = func = wasmTable.get(funcPtr);
      if (Asyncify.isAsyncExport(func)) {
        wasmTableMirror[funcPtr] = func = Asyncify.makeAsyncFunction(func);
      }
    }
    return func;
  };
  var WebGPU = { Internals: { jsObjects: [], jsObjectInsert: (ptr, jsObject) => {
    ptr >>>= 0;
    WebGPU.Internals.jsObjects[ptr] = jsObject;
  }, bufferOnUnmaps: [], futures: [], futureInsert: (futureId, promise) => {
    WebGPU.Internals.futures[futureId] = new Promise((resolve) => promise.finally(() => resolve(futureId)));
  } }, getJsObject: (ptr) => {
    if (!ptr)
      return;
    ptr >>>= 0;
    return WebGPU.Internals.jsObjects[ptr];
  }, importJsAdapter: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateAdapter(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsBindGroup: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateBindGroup(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsBindGroupLayout: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateBindGroupLayout(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsBuffer: (buffer, parentPtr = 0) => {
    assert(buffer.mapState === "unmapped");
    var bufferPtr = _emwgpuImportBuffer(parentPtr);
    WebGPU.Internals.jsObjectInsert(bufferPtr, buffer);
    return bufferPtr;
  }, importJsCommandBuffer: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateCommandBuffer(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsCommandEncoder: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateCommandEncoder(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsComputePassEncoder: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateComputePassEncoder(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsComputePipeline: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateComputePipeline(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsDevice: (device, parentPtr = 0) => {
    var queuePtr = _emwgpuCreateQueue(parentPtr);
    var devicePtr = _emwgpuCreateDevice(parentPtr, queuePtr);
    WebGPU.Internals.jsObjectInsert(queuePtr, device.queue);
    WebGPU.Internals.jsObjectInsert(devicePtr, device);
    return devicePtr;
  }, importJsExternalTexture: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateExternalTexture(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsPipelineLayout: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreatePipelineLayout(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsQuerySet: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateQuerySet(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsQueue: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateQueue(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsRenderBundle: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateRenderBundle(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsRenderBundleEncoder: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateRenderBundleEncoder(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsRenderPassEncoder: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateRenderPassEncoder(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsRenderPipeline: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateRenderPipeline(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsSampler: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateSampler(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsShaderModule: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateShaderModule(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsSurface: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateSurface(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsTexture: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateTexture(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, importJsTextureView: (obj, parentPtr = 0) => {
    var ptr = _emwgpuCreateTextureView(parentPtr);
    WebGPU.Internals.jsObjects[ptr] = obj;
    return ptr;
  }, errorCallback: (callback, type, message, userdata) => {
    var sp = stackSave();
    var messagePtr = stringToUTF8OnStack(message);
    getWasmTableEntry(callback)(type, messagePtr, userdata);
    stackRestore(sp);
  }, iterateExtensions: (root, handlers) => {
    for (var ptr = HEAPU32[root >>> 2 >>> 0];ptr; ptr = HEAPU32[ptr >>> 2 >>> 0]) {
      var sType = HEAP32[ptr + 4 >>> 2 >>> 0];
      var handler = handlers[sType](ptr);
    }
  }, setStringView: (ptr, data, length) => {
    HEAPU32[ptr >>> 2 >>> 0] = data;
    HEAPU32[ptr + 4 >>> 2 >>> 0] = length;
  }, makeStringFromStringView: (stringViewPtr) => {
    var ptr = HEAPU32[stringViewPtr >>> 2 >>> 0];
    var length = HEAPU32[stringViewPtr + 4 >>> 2 >>> 0];
    return UTF8ToString(ptr, length);
  }, makeStringFromOptionalStringView: (stringViewPtr) => {
    var ptr = HEAPU32[stringViewPtr >>> 2 >>> 0];
    var length = HEAPU32[stringViewPtr + 4 >>> 2 >>> 0];
    if (!ptr) {
      if (length === 0) {
        return "";
      }
      return;
    }
    return UTF8ToString(ptr, length);
  }, makeColor: (ptr) => ({ r: HEAPF64[ptr >>> 3 >>> 0], g: HEAPF64[ptr + 8 >>> 3 >>> 0], b: HEAPF64[ptr + 16 >>> 3 >>> 0], a: HEAPF64[ptr + 24 >>> 3 >>> 0] }), makeExtent3D: (ptr) => ({ width: HEAPU32[ptr >>> 2 >>> 0], height: HEAPU32[ptr + 4 >>> 2 >>> 0], depthOrArrayLayers: HEAPU32[ptr + 8 >>> 2 >>> 0] }), makeOrigin3D: (ptr) => ({ x: HEAPU32[ptr >>> 2 >>> 0], y: HEAPU32[ptr + 4 >>> 2 >>> 0], z: HEAPU32[ptr + 8 >>> 2 >>> 0] }), makeTexelCopyTextureInfo: (ptr) => ({ texture: WebGPU.getJsObject(HEAPU32[ptr >>> 2 >>> 0]), mipLevel: HEAPU32[ptr + 4 >>> 2 >>> 0], origin: WebGPU.makeOrigin3D(ptr + 8), aspect: WebGPU.TextureAspect[HEAP32[ptr + 20 >>> 2 >>> 0]] }), makeTexelCopyBufferLayout: (ptr) => {
    var bytesPerRow = HEAPU32[ptr + 8 >>> 2 >>> 0];
    var rowsPerImage = HEAPU32[ptr + 12 >>> 2 >>> 0];
    return { offset: readI53FromI64(ptr), bytesPerRow: bytesPerRow === 4294967295 ? undefined : bytesPerRow, rowsPerImage: rowsPerImage === 4294967295 ? undefined : rowsPerImage };
  }, makeTexelCopyBufferInfo: (ptr) => {
    var layoutPtr = ptr + 0;
    var bufferCopyView = WebGPU.makeTexelCopyBufferLayout(layoutPtr);
    bufferCopyView["buffer"] = WebGPU.getJsObject(HEAPU32[ptr + 16 >>> 2 >>> 0]);
    return bufferCopyView;
  }, makePassTimestampWrites: (ptr) => {
    if (ptr === 0)
      return;
    return { querySet: WebGPU.getJsObject(HEAPU32[ptr + 4 >>> 2 >>> 0]), beginningOfPassWriteIndex: HEAPU32[ptr + 8 >>> 2 >>> 0], endOfPassWriteIndex: HEAPU32[ptr + 12 >>> 2 >>> 0] };
  }, makePipelineConstants: (constantCount, constantsPtr) => {
    if (!constantCount)
      return;
    var constants = {};
    for (var i = 0;i < constantCount; ++i) {
      var entryPtr = constantsPtr + 24 * i;
      var key = WebGPU.makeStringFromStringView(entryPtr + 4);
      constants[key] = HEAPF64[entryPtr + 16 >>> 3 >>> 0];
    }
    return constants;
  }, makePipelineLayout: (layoutPtr) => {
    if (!layoutPtr)
      return "auto";
    return WebGPU.getJsObject(layoutPtr);
  }, makeComputeState: (ptr) => {
    if (!ptr)
      return;
    var desc = { module: WebGPU.getJsObject(HEAPU32[ptr + 4 >>> 2 >>> 0]), constants: WebGPU.makePipelineConstants(HEAPU32[ptr + 16 >>> 2 >>> 0], HEAPU32[ptr + 20 >>> 2 >>> 0]), entryPoint: WebGPU.makeStringFromOptionalStringView(ptr + 8) };
    return desc;
  }, makeComputePipelineDesc: (descriptor) => {
    var desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4), layout: WebGPU.makePipelineLayout(HEAPU32[descriptor + 12 >>> 2 >>> 0]), compute: WebGPU.makeComputeState(descriptor + 16) };
    return desc;
  }, makeRenderPipelineDesc: (descriptor) => {
    function makePrimitiveState(psPtr) {
      if (!psPtr)
        return;
      return { topology: WebGPU.PrimitiveTopology[HEAP32[psPtr + 4 >>> 2 >>> 0]], stripIndexFormat: WebGPU.IndexFormat[HEAP32[psPtr + 8 >>> 2 >>> 0]], frontFace: WebGPU.FrontFace[HEAP32[psPtr + 12 >>> 2 >>> 0]], cullMode: WebGPU.CullMode[HEAP32[psPtr + 16 >>> 2 >>> 0]], unclippedDepth: !!HEAPU32[psPtr + 20 >>> 2 >>> 0] };
    }
    function makeBlendComponent(bdPtr) {
      if (!bdPtr)
        return;
      return { operation: WebGPU.BlendOperation[HEAP32[bdPtr >>> 2 >>> 0]], srcFactor: WebGPU.BlendFactor[HEAP32[bdPtr + 4 >>> 2 >>> 0]], dstFactor: WebGPU.BlendFactor[HEAP32[bdPtr + 8 >>> 2 >>> 0]] };
    }
    function makeBlendState(bsPtr) {
      if (!bsPtr)
        return;
      return { alpha: makeBlendComponent(bsPtr + 12), color: makeBlendComponent(bsPtr + 0) };
    }
    function makeColorState(csPtr) {
      var format = WebGPU.TextureFormat[HEAP32[csPtr + 4 >>> 2 >>> 0]];
      return format ? { format, blend: makeBlendState(HEAPU32[csPtr + 8 >>> 2 >>> 0]), writeMask: HEAPU32[csPtr + 16 >>> 2 >>> 0] } : undefined;
    }
    function makeColorStates(count, csArrayPtr) {
      var states = [];
      for (var i = 0;i < count; ++i) {
        states.push(makeColorState(csArrayPtr + 24 * i));
      }
      return states;
    }
    function makeStencilStateFace(ssfPtr) {
      return { compare: WebGPU.CompareFunction[HEAP32[ssfPtr >>> 2 >>> 0]], failOp: WebGPU.StencilOperation[HEAP32[ssfPtr + 4 >>> 2 >>> 0]], depthFailOp: WebGPU.StencilOperation[HEAP32[ssfPtr + 8 >>> 2 >>> 0]], passOp: WebGPU.StencilOperation[HEAP32[ssfPtr + 12 >>> 2 >>> 0]] };
    }
    function makeDepthStencilState(dssPtr) {
      if (!dssPtr)
        return;
      return { format: WebGPU.TextureFormat[HEAP32[dssPtr + 4 >>> 2 >>> 0]], depthWriteEnabled: !!HEAPU32[dssPtr + 8 >>> 2 >>> 0], depthCompare: WebGPU.CompareFunction[HEAP32[dssPtr + 12 >>> 2 >>> 0]], stencilFront: makeStencilStateFace(dssPtr + 16), stencilBack: makeStencilStateFace(dssPtr + 32), stencilReadMask: HEAPU32[dssPtr + 48 >>> 2 >>> 0], stencilWriteMask: HEAPU32[dssPtr + 52 >>> 2 >>> 0], depthBias: HEAP32[dssPtr + 56 >>> 2 >>> 0], depthBiasSlopeScale: HEAPF32[dssPtr + 60 >>> 2 >>> 0], depthBiasClamp: HEAPF32[dssPtr + 64 >>> 2 >>> 0] };
    }
    function makeVertexAttribute(vaPtr) {
      return { format: WebGPU.VertexFormat[HEAP32[vaPtr + 4 >>> 2 >>> 0]], offset: readI53FromI64(vaPtr + 8), shaderLocation: HEAPU32[vaPtr + 16 >>> 2 >>> 0] };
    }
    function makeVertexAttributes(count, vaArrayPtr) {
      var vas = [];
      for (var i = 0;i < count; ++i) {
        vas.push(makeVertexAttribute(vaArrayPtr + i * 24));
      }
      return vas;
    }
    function makeVertexBuffer(vbPtr) {
      if (!vbPtr)
        return;
      var stepMode = WebGPU.VertexStepMode[HEAP32[vbPtr + 4 >>> 2 >>> 0]];
      var attributeCount = HEAPU32[vbPtr + 16 >>> 2 >>> 0];
      if (!stepMode && !attributeCount) {
        return null;
      }
      return { arrayStride: readI53FromI64(vbPtr + 8), stepMode, attributes: makeVertexAttributes(attributeCount, HEAPU32[vbPtr + 20 >>> 2 >>> 0]) };
    }
    function makeVertexBuffers(count, vbArrayPtr) {
      if (!count)
        return;
      var vbs = [];
      for (var i = 0;i < count; ++i) {
        vbs.push(makeVertexBuffer(vbArrayPtr + i * 24));
      }
      return vbs;
    }
    function makeVertexState(viPtr) {
      if (!viPtr)
        return;
      var desc2 = { module: WebGPU.getJsObject(HEAPU32[viPtr + 4 >>> 2 >>> 0]), constants: WebGPU.makePipelineConstants(HEAPU32[viPtr + 16 >>> 2 >>> 0], HEAPU32[viPtr + 20 >>> 2 >>> 0]), buffers: makeVertexBuffers(HEAPU32[viPtr + 24 >>> 2 >>> 0], HEAPU32[viPtr + 28 >>> 2 >>> 0]), entryPoint: WebGPU.makeStringFromOptionalStringView(viPtr + 8) };
      return desc2;
    }
    function makeMultisampleState(msPtr) {
      if (!msPtr)
        return;
      return { count: HEAPU32[msPtr + 4 >>> 2 >>> 0], mask: HEAPU32[msPtr + 8 >>> 2 >>> 0], alphaToCoverageEnabled: !!HEAPU32[msPtr + 12 >>> 2 >>> 0] };
    }
    function makeFragmentState(fsPtr) {
      if (!fsPtr)
        return;
      var desc2 = { module: WebGPU.getJsObject(HEAPU32[fsPtr + 4 >>> 2 >>> 0]), constants: WebGPU.makePipelineConstants(HEAPU32[fsPtr + 16 >>> 2 >>> 0], HEAPU32[fsPtr + 20 >>> 2 >>> 0]), targets: makeColorStates(HEAPU32[fsPtr + 24 >>> 2 >>> 0], HEAPU32[fsPtr + 28 >>> 2 >>> 0]), entryPoint: WebGPU.makeStringFromOptionalStringView(fsPtr + 8) };
      return desc2;
    }
    var desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4), layout: WebGPU.makePipelineLayout(HEAPU32[descriptor + 12 >>> 2 >>> 0]), vertex: makeVertexState(descriptor + 16), primitive: makePrimitiveState(descriptor + 48), depthStencil: makeDepthStencilState(HEAPU32[descriptor + 72 >>> 2 >>> 0]), multisample: makeMultisampleState(descriptor + 76), fragment: makeFragmentState(HEAPU32[descriptor + 92 >>> 2 >>> 0]) };
    return desc;
  }, fillLimitStruct: (limits, limitsOutPtr) => {
    var nextInChainPtr = HEAPU32[limitsOutPtr >>> 2 >>> 0];
    function setLimitValueU32(name, basePtr, limitOffset, fallbackValue = 0) {
      var limitValue = limits[name] ?? fallbackValue;
      HEAPU32[basePtr + limitOffset >>> 2 >>> 0] = limitValue;
    }
    function setLimitValueU64(name, basePtr, limitOffset, fallbackValue = 0) {
      var limitValue = limits[name] ?? fallbackValue;
      writeI53ToI64(basePtr + limitOffset, limitValue);
    }
    setLimitValueU32("maxTextureDimension1D", limitsOutPtr, 4);
    setLimitValueU32("maxTextureDimension2D", limitsOutPtr, 8);
    setLimitValueU32("maxTextureDimension3D", limitsOutPtr, 12);
    setLimitValueU32("maxTextureArrayLayers", limitsOutPtr, 16);
    setLimitValueU32("maxBindGroups", limitsOutPtr, 20);
    setLimitValueU32("maxBindGroupsPlusVertexBuffers", limitsOutPtr, 24);
    setLimitValueU32("maxBindingsPerBindGroup", limitsOutPtr, 28);
    setLimitValueU32("maxDynamicUniformBuffersPerPipelineLayout", limitsOutPtr, 32);
    setLimitValueU32("maxDynamicStorageBuffersPerPipelineLayout", limitsOutPtr, 36);
    setLimitValueU32("maxSampledTexturesPerShaderStage", limitsOutPtr, 40);
    setLimitValueU32("maxSamplersPerShaderStage", limitsOutPtr, 44);
    setLimitValueU32("maxStorageBuffersPerShaderStage", limitsOutPtr, 48);
    setLimitValueU32("maxStorageTexturesPerShaderStage", limitsOutPtr, 52);
    setLimitValueU32("maxUniformBuffersPerShaderStage", limitsOutPtr, 56);
    setLimitValueU32("minUniformBufferOffsetAlignment", limitsOutPtr, 80);
    setLimitValueU32("minStorageBufferOffsetAlignment", limitsOutPtr, 84);
    setLimitValueU64("maxUniformBufferBindingSize", limitsOutPtr, 64);
    setLimitValueU64("maxStorageBufferBindingSize", limitsOutPtr, 72);
    setLimitValueU32("maxVertexBuffers", limitsOutPtr, 88);
    setLimitValueU64("maxBufferSize", limitsOutPtr, 96);
    setLimitValueU32("maxVertexAttributes", limitsOutPtr, 104);
    setLimitValueU32("maxVertexBufferArrayStride", limitsOutPtr, 108);
    setLimitValueU32("maxInterStageShaderVariables", limitsOutPtr, 112);
    setLimitValueU32("maxColorAttachments", limitsOutPtr, 116);
    setLimitValueU32("maxColorAttachmentBytesPerSample", limitsOutPtr, 120);
    setLimitValueU32("maxComputeWorkgroupStorageSize", limitsOutPtr, 124);
    setLimitValueU32("maxComputeInvocationsPerWorkgroup", limitsOutPtr, 128);
    setLimitValueU32("maxComputeWorkgroupSizeX", limitsOutPtr, 132);
    setLimitValueU32("maxComputeWorkgroupSizeY", limitsOutPtr, 136);
    setLimitValueU32("maxComputeWorkgroupSizeZ", limitsOutPtr, 140);
    setLimitValueU32("maxComputeWorkgroupsPerDimension", limitsOutPtr, 144);
    setLimitValueU32("maxImmediateSize", limitsOutPtr, 148);
    if (nextInChainPtr !== 0) {
      var sType = HEAP32[nextInChainPtr + 4 >>> 2 >>> 0];
      var compatibilityModeLimitsPtr = nextInChainPtr;
      setLimitValueU32("maxStorageBuffersInVertexStage", compatibilityModeLimitsPtr, 8, limits.maxStorageBuffersPerShaderStage);
      setLimitValueU32("maxStorageBuffersInFragmentStage", compatibilityModeLimitsPtr, 16, limits.maxStorageBuffersPerShaderStage);
      setLimitValueU32("maxStorageTexturesInVertexStage", compatibilityModeLimitsPtr, 12, limits.maxStorageTexturesPerShaderStage);
      setLimitValueU32("maxStorageTexturesInFragmentStage", compatibilityModeLimitsPtr, 20, limits.maxStorageTexturesPerShaderStage);
    }
  }, fillAdapterInfoStruct: (info, infoStruct) => {
    HEAPU32[infoStruct + 52 >>> 2 >>> 0] = info.subgroupMinSize;
    HEAPU32[infoStruct + 56 >>> 2 >>> 0] = info.subgroupMaxSize;
    var strs = info.vendor + info.architecture + info.device + info.description;
    var strPtr = stringToNewUTF8(strs);
    var vendorLen = lengthBytesUTF8(info.vendor);
    WebGPU.setStringView(infoStruct + 4, strPtr, vendorLen);
    strPtr += vendorLen;
    var architectureLen = lengthBytesUTF8(info.architecture);
    WebGPU.setStringView(infoStruct + 12, strPtr, architectureLen);
    strPtr += architectureLen;
    var deviceLen = lengthBytesUTF8(info.device);
    WebGPU.setStringView(infoStruct + 20, strPtr, deviceLen);
    strPtr += deviceLen;
    var descriptionLen = lengthBytesUTF8(info.description);
    WebGPU.setStringView(infoStruct + 28, strPtr, descriptionLen);
    strPtr += descriptionLen;
    HEAP32[infoStruct + 36 >>> 2 >>> 0] = 2;
    var adapterType = info.isFallbackAdapter ? 3 : 4;
    HEAP32[infoStruct + 40 >>> 2 >>> 0] = adapterType;
    HEAPU32[infoStruct + 44 >>> 2 >>> 0] = 0;
    HEAPU32[infoStruct + 48 >>> 2 >>> 0] = 0;
  }, AddressMode: [, "clamp-to-edge", "repeat", "mirror-repeat"], BlendFactor: [, "zero", "one", "src", "one-minus-src", "src-alpha", "one-minus-src-alpha", "dst", "one-minus-dst", "dst-alpha", "one-minus-dst-alpha", "src-alpha-saturated", "constant", "one-minus-constant", "src1", "one-minus-src1", "src1-alpha", "one-minus-src1-alpha"], BlendOperation: [, "add", "subtract", "reverse-subtract", "min", "max"], BufferBindingType: [, , "uniform", "storage", "read-only-storage"], BufferMapState: [, "unmapped", "pending", "mapped"], CompareFunction: [, "never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"], CompilationInfoRequestStatus: [, "success", "callback-cancelled"], ComponentSwizzle: [, "0", "1", "r", "g", "b", "a"], CompositeAlphaMode: [, "opaque", "premultiplied", "unpremultiplied", "inherit"], CullMode: [, "none", "front", "back"], ErrorFilter: [, "validation", "out-of-memory", "internal"], FeatureLevel: [, "compatibility", "core"], FeatureName: { 1: "core-features-and-limits", 2: "depth-clip-control", 3: "depth32float-stencil8", 4: "texture-compression-bc", 5: "texture-compression-bc-sliced-3d", 6: "texture-compression-etc2", 7: "texture-compression-astc", 8: "texture-compression-astc-sliced-3d", 9: "timestamp-query", 10: "indirect-first-instance", 11: "shader-f16", 12: "rg11b10ufloat-renderable", 13: "bgra8unorm-storage", 14: "float32-filterable", 15: "float32-blendable", 16: "clip-distances", 17: "dual-source-blending", 18: "subgroups", 19: "texture-formats-tier1", 20: "texture-formats-tier2", 21: "primitive-index", 22: "texture-component-swizzle", 327692: "chromium-experimental-unorm16-texture-formats", 327729: "chromium-experimental-multi-draw-indirect" }, FilterMode: [, "nearest", "linear"], FrontFace: [, "ccw", "cw"], IndexFormat: [, "uint16", "uint32"], InstanceFeatureName: [, "timed-wait-any", "shader-source-spirv", "multiple-devices-per-adapter"], LoadOp: [, "load", "clear"], MipmapFilterMode: [, "nearest", "linear"], OptionalBool: ["false", "true"], PowerPreference: [, "low-power", "high-performance"], PredefinedColorSpace: [, "srgb", "display-p3"], PrimitiveTopology: [, "point-list", "line-list", "line-strip", "triangle-list", "triangle-strip"], QueryType: [, "occlusion", "timestamp"], SamplerBindingType: [, , "filtering", "non-filtering", "comparison"], Status: [, "success", "error"], StencilOperation: [, "keep", "zero", "replace", "invert", "increment-clamp", "decrement-clamp", "increment-wrap", "decrement-wrap"], StorageTextureAccess: [, , "write-only", "read-only", "read-write"], StoreOp: [, "store", "discard"], SurfaceGetCurrentTextureStatus: [, "success-optimal", "success-suboptimal", "timeout", "outdated", "lost", "error"], TextureAspect: [, "all", "stencil-only", "depth-only"], TextureDimension: [, "1d", "2d", "3d"], TextureFormat: [, "r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint", "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32float", "r32uint", "r32sint", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm", "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rgb9e5ufloat", "rg32float", "rg32uint", "rg32sint", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float", "rgba32float", "rgba32uint", "rgba32sint", "stencil8", "depth16unorm", "depth24plus", "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm", "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm", "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm", "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb", "etc2-rgb8unorm", "etc2-rgb8unorm-srgb", "etc2-rgb8a1unorm", "etc2-rgb8a1unorm-srgb", "etc2-rgba8unorm", "etc2-rgba8unorm-srgb", "eac-r11unorm", "eac-r11snorm", "eac-rg11unorm", "eac-rg11snorm", "astc-4x4-unorm", "astc-4x4-unorm-srgb", "astc-5x4-unorm", "astc-5x4-unorm-srgb", "astc-5x5-unorm", "astc-5x5-unorm-srgb", "astc-6x5-unorm", "astc-6x5-unorm-srgb", "astc-6x6-unorm", "astc-6x6-unorm-srgb", "astc-8x5-unorm", "astc-8x5-unorm-srgb", "astc-8x6-unorm", "astc-8x6-unorm-srgb", "astc-8x8-unorm", "astc-8x8-unorm-srgb", "astc-10x5-unorm", "astc-10x5-unorm-srgb", "astc-10x6-unorm", "astc-10x6-unorm-srgb", "astc-10x8-unorm", "astc-10x8-unorm-srgb", "astc-10x10-unorm", "astc-10x10-unorm-srgb", "astc-12x10-unorm", "astc-12x10-unorm-srgb", "astc-12x12-unorm", "astc-12x12-unorm-srgb"], TextureSampleType: [, , "float", "unfilterable-float", "depth", "sint", "uint"], TextureViewDimension: [, "1d", "2d", "2d-array", "cube", "cube-array", "3d"], ToneMappingMode: [, "standard", "extended"], VertexFormat: [, "uint8", "uint8x2", "uint8x4", "sint8", "sint8x2", "sint8x4", "unorm8", "unorm8x2", "unorm8x4", "snorm8", "snorm8x2", "snorm8x4", "uint16", "uint16x2", "uint16x4", "sint16", "sint16x2", "sint16x4", "unorm16", "unorm16x2", "unorm16x4", "snorm16", "snorm16x2", "snorm16x4", "float16", "float16x2", "float16x4", "float32", "float32x2", "float32x3", "float32x4", "uint32", "uint32x2", "uint32x3", "uint32x4", "sint32", "sint32x2", "sint32x3", "sint32x4", "unorm10-10-10-2", "unorm8x4-bgra"], VertexStepMode: [, "vertex", "instance"], WGSLLanguageFeatureName: [, "readonly_and_readwrite_storage_textures", "packed_4x8_integer_dot_product", "unrestricted_pointer_parameters", "pointer_composite_access", "uniform_buffer_standard_layout", "subgroup_id", "texture_and_sampler_let", "subgroup_uniformity", "texture_formats_tier1", "linear_indexing"] };
  var emwgpuStringToInt_DeviceLostReason = { undefined: 1, unknown: 1, destroyed: 2 };
  var handleException = (e) => {
    if (e instanceof ExitStatus || e == "unwind") {
      return EXITSTATUS;
    }
    quit_(1, e);
  };
  var runtimeKeepaliveCounter = 0;
  var keepRuntimeAlive = () => noExitRuntime || runtimeKeepaliveCounter > 0;
  var _proc_exit = (code) => {
    EXITSTATUS = code;
    if (!keepRuntimeAlive()) {
      Module2["onExit"]?.(code);
      ABORT = true;
    }
    quit_(code, new ExitStatus(code));
  };
  var exitJS = (status, implicit) => {
    EXITSTATUS = status;
    _proc_exit(status);
  };
  var _exit = exitJS;
  var maybeExit = () => {
    if (!keepRuntimeAlive()) {
      try {
        _exit(EXITSTATUS);
      } catch (e) {
        handleException(e);
      }
    }
  };
  var callUserCallback = (func) => {
    if (ABORT) {
      return;
    }
    try {
      return func();
    } catch (e) {
      handleException(e);
    } finally {
      maybeExit();
    }
  };
  function _emwgpuAdapterRequestDevice(adapterPtr, futureId, deviceLostFutureId, devicePtr, queuePtr, descriptor) {
    adapterPtr >>>= 0;
    futureId = bigintToI53Checked(futureId);
    deviceLostFutureId = bigintToI53Checked(deviceLostFutureId);
    devicePtr >>>= 0;
    queuePtr >>>= 0;
    descriptor >>>= 0;
    var adapter = WebGPU.getJsObject(adapterPtr);
    var desc = {};
    if (descriptor) {
      var requiredFeatureCount = HEAPU32[descriptor + 12 >>> 2 >>> 0];
      if (requiredFeatureCount) {
        var requiredFeaturesPtr = HEAPU32[descriptor + 16 >>> 2 >>> 0];
        desc["requiredFeatures"] = Array.from(HEAPU32.subarray(requiredFeaturesPtr >>> 2 >>> 0, requiredFeaturesPtr + requiredFeatureCount * 4 >>> 2 >>> 0), (feature) => WebGPU.FeatureName[feature]);
      }
      var limitsPtr = HEAPU32[descriptor + 20 >>> 2 >>> 0];
      if (limitsPtr) {
        let setLimitU32IfDefined = function(name, basePtr, limitOffset, ignoreIfZero = false) {
          var ptr = basePtr + limitOffset;
          var value = HEAPU32[ptr >>> 2 >>> 0];
          if (value != 4294967295 && (!ignoreIfZero || value != 0)) {
            requiredLimits[name] = value;
          }
        }, setLimitU64IfDefined = function(name, basePtr, limitOffset) {
          var ptr = basePtr + limitOffset;
          var limitPart1 = HEAPU32[ptr >>> 2 >>> 0];
          var limitPart2 = HEAPU32[ptr + 4 >>> 2 >>> 0];
          if (limitPart1 != 4294967295 || limitPart2 != 4294967295) {
            requiredLimits[name] = readI53FromI64(ptr);
          }
        };
        var nextInChainPtr = HEAPU32[limitsPtr >>> 2 >>> 0];
        var requiredLimits = {};
        setLimitU32IfDefined("maxTextureDimension1D", limitsPtr, 4);
        setLimitU32IfDefined("maxTextureDimension2D", limitsPtr, 8);
        setLimitU32IfDefined("maxTextureDimension3D", limitsPtr, 12);
        setLimitU32IfDefined("maxTextureArrayLayers", limitsPtr, 16);
        setLimitU32IfDefined("maxBindGroups", limitsPtr, 20);
        setLimitU32IfDefined("maxBindGroupsPlusVertexBuffers", limitsPtr, 24);
        setLimitU32IfDefined("maxBindingsPerBindGroup", limitsPtr, 28);
        setLimitU32IfDefined("maxDynamicUniformBuffersPerPipelineLayout", limitsPtr, 32);
        setLimitU32IfDefined("maxDynamicStorageBuffersPerPipelineLayout", limitsPtr, 36);
        setLimitU32IfDefined("maxSampledTexturesPerShaderStage", limitsPtr, 40);
        setLimitU32IfDefined("maxSamplersPerShaderStage", limitsPtr, 44);
        setLimitU32IfDefined("maxStorageBuffersPerShaderStage", limitsPtr, 48);
        setLimitU32IfDefined("maxStorageTexturesPerShaderStage", limitsPtr, 52);
        setLimitU32IfDefined("maxUniformBuffersPerShaderStage", limitsPtr, 56);
        setLimitU32IfDefined("minUniformBufferOffsetAlignment", limitsPtr, 80);
        setLimitU32IfDefined("minStorageBufferOffsetAlignment", limitsPtr, 84);
        setLimitU64IfDefined("maxUniformBufferBindingSize", limitsPtr, 64);
        setLimitU64IfDefined("maxStorageBufferBindingSize", limitsPtr, 72);
        setLimitU32IfDefined("maxVertexBuffers", limitsPtr, 88);
        setLimitU64IfDefined("maxBufferSize", limitsPtr, 96);
        setLimitU32IfDefined("maxVertexAttributes", limitsPtr, 104);
        setLimitU32IfDefined("maxVertexBufferArrayStride", limitsPtr, 108);
        setLimitU32IfDefined("maxInterStageShaderVariables", limitsPtr, 112);
        setLimitU32IfDefined("maxColorAttachments", limitsPtr, 116);
        setLimitU32IfDefined("maxColorAttachmentBytesPerSample", limitsPtr, 120);
        setLimitU32IfDefined("maxComputeWorkgroupStorageSize", limitsPtr, 124);
        setLimitU32IfDefined("maxComputeInvocationsPerWorkgroup", limitsPtr, 128);
        setLimitU32IfDefined("maxComputeWorkgroupSizeX", limitsPtr, 132);
        setLimitU32IfDefined("maxComputeWorkgroupSizeY", limitsPtr, 136);
        setLimitU32IfDefined("maxComputeWorkgroupSizeZ", limitsPtr, 140);
        setLimitU32IfDefined("maxComputeWorkgroupsPerDimension", limitsPtr, 144);
        setLimitU32IfDefined("maxImmediateSize", limitsPtr, 148, true);
        if (nextInChainPtr !== 0) {
          var sType = HEAP32[nextInChainPtr + 4 >>> 2 >>> 0];
          var compatibilityModeLimitsPtr = nextInChainPtr;
          if ("maxStorageBuffersInVertexStage" in GPUSupportedLimits.prototype) {
            setLimitU32IfDefined("maxStorageBuffersInVertexStage", compatibilityModeLimitsPtr, 8);
            setLimitU32IfDefined("maxStorageTexturesInVertexStage", compatibilityModeLimitsPtr, 12);
            setLimitU32IfDefined("maxStorageBuffersInFragmentStage", compatibilityModeLimitsPtr, 16);
            setLimitU32IfDefined("maxStorageTexturesInFragmentStage", compatibilityModeLimitsPtr, 20);
          }
        }
        desc["requiredLimits"] = requiredLimits;
      }
      var defaultQueuePtr = HEAPU32[descriptor + 24 >>> 2 >>> 0];
      if (defaultQueuePtr) {
        var defaultQueueDesc = { label: WebGPU.makeStringFromOptionalStringView(defaultQueuePtr + 4) };
        desc["defaultQueue"] = defaultQueueDesc;
      }
      desc["label"] = WebGPU.makeStringFromOptionalStringView(descriptor + 4);
    }
    WebGPU.Internals.futureInsert(futureId, adapter.requestDevice(desc).then((device) => {
      callUserCallback(() => {
        WebGPU.Internals.jsObjectInsert(queuePtr, device.queue);
        WebGPU.Internals.jsObjectInsert(devicePtr, device);
        WebGPU.Internals.futureInsert(deviceLostFutureId, device.lost.then((info) => {
          callUserCallback(() => {
            device.onuncapturederror = (ev) => {};
            var sp = stackSave();
            var messagePtr = stringToUTF8OnStack(info.message);
            _emwgpuOnDeviceLostCompleted(deviceLostFutureId, emwgpuStringToInt_DeviceLostReason[info.reason], messagePtr);
            stackRestore(sp);
          });
        }));
        device.onuncapturederror = (ev) => {
          var type = 5;
          if (ev.error instanceof GPUValidationError)
            type = 2;
          else if (ev.error instanceof GPUOutOfMemoryError)
            type = 3;
          else if (ev.error instanceof GPUInternalError)
            type = 4;
          var sp = stackSave();
          var messagePtr = stringToUTF8OnStack(ev.error.message);
          _emwgpuOnUncapturedError(devicePtr, type, messagePtr);
          stackRestore(sp);
        };
        _emwgpuOnRequestDeviceCompleted(futureId, 1, devicePtr, 0);
      });
    }, (ex) => {
      callUserCallback(() => {
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(ex.message);
        _emwgpuOnRequestDeviceCompleted(futureId, 3, devicePtr, messagePtr);
        if (deviceLostFutureId) {
          _emwgpuOnDeviceLostCompleted(deviceLostFutureId, 4, messagePtr);
        }
        stackRestore(sp);
      });
    }));
  }
  function _emwgpuBufferDestroy(bufferPtr) {
    bufferPtr >>>= 0;
    var buffer = WebGPU.getJsObject(bufferPtr);
    var onUnmap = WebGPU.Internals.bufferOnUnmaps[bufferPtr];
    if (onUnmap) {
      for (var i = 0;i < onUnmap.length; ++i) {
        onUnmap[i]();
      }
      delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
    }
    buffer.destroy();
  }
  var warnOnce = (text) => {
    warnOnce.shown ||= {};
    if (!warnOnce.shown[text]) {
      warnOnce.shown[text] = 1;
      err(text);
    }
  };
  function _emwgpuBufferGetConstMappedRange(bufferPtr, offset, size) {
    bufferPtr >>>= 0;
    offset >>>= 0;
    size >>>= 0;
    var buffer = WebGPU.getJsObject(bufferPtr);
    if (size == 4294967295)
      size = undefined;
    var mapped;
    try {
      mapped = buffer.getMappedRange(offset, size);
    } catch (ex) {
      return 0;
    }
    var data = _memalign(16, mapped.byteLength);
    HEAPU8.set(new Uint8Array(mapped), data >>> 0);
    WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(() => _free(data));
    return data;
  }
  var _emwgpuBufferMapAsync = function(bufferPtr, futureId, mode, offset, size) {
    bufferPtr >>>= 0;
    futureId = bigintToI53Checked(futureId);
    mode = bigintToI53Checked(mode);
    offset >>>= 0;
    size >>>= 0;
    var buffer = WebGPU.getJsObject(bufferPtr);
    WebGPU.Internals.bufferOnUnmaps[bufferPtr] = [];
    if (size == 4294967295)
      size = undefined;
    WebGPU.Internals.futureInsert(futureId, buffer.mapAsync(mode, offset, size).then(() => {
      callUserCallback(() => {
        _emwgpuOnMapAsyncCompleted(futureId, 1, 0);
      });
    }, (ex) => {
      callUserCallback(() => {
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(ex.message);
        var status = ex.name === "AbortError" ? 4 : ex.name === "OperationError" ? 3 : 0;
        _emwgpuOnMapAsyncCompleted(futureId, status, messagePtr);
        delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      });
    }));
  };
  function _emwgpuBufferUnmap(bufferPtr) {
    bufferPtr >>>= 0;
    var buffer = WebGPU.getJsObject(bufferPtr);
    var onUnmap = WebGPU.Internals.bufferOnUnmaps[bufferPtr];
    if (!onUnmap) {
      return;
    }
    for (var i = 0;i < onUnmap.length; ++i) {
      onUnmap[i]();
    }
    delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
    buffer.unmap();
  }
  function _emwgpuDelete(ptr) {
    ptr >>>= 0;
    delete WebGPU.Internals.jsObjects[ptr];
  }
  function _emwgpuDeviceCreateBuffer(devicePtr, descriptor, bufferPtr) {
    devicePtr >>>= 0;
    descriptor >>>= 0;
    bufferPtr >>>= 0;
    var mappedAtCreation = !!HEAPU32[descriptor + 32 >>> 2 >>> 0];
    var desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4), usage: HEAPU32[descriptor + 16 >>> 2 >>> 0], size: readI53FromI64(descriptor + 24), mappedAtCreation };
    var device = WebGPU.getJsObject(devicePtr);
    var buffer;
    try {
      buffer = device.createBuffer(desc);
    } catch (ex) {
      return false;
    }
    WebGPU.Internals.jsObjectInsert(bufferPtr, buffer);
    if (mappedAtCreation) {
      WebGPU.Internals.bufferOnUnmaps[bufferPtr] = [];
    }
    return true;
  }
  function _emwgpuDeviceCreateShaderModule(devicePtr, descriptor, shaderModulePtr) {
    devicePtr >>>= 0;
    descriptor >>>= 0;
    shaderModulePtr >>>= 0;
    var nextInChainPtr = HEAPU32[descriptor >>> 2 >>> 0];
    var sType = HEAP32[nextInChainPtr + 4 >>> 2 >>> 0];
    var desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4), code: "" };
    switch (sType) {
      case 2: {
        desc["code"] = WebGPU.makeStringFromStringView(nextInChainPtr + 8);
        break;
      }
    }
    var device = WebGPU.getJsObject(devicePtr);
    WebGPU.Internals.jsObjectInsert(shaderModulePtr, device.createShaderModule(desc));
  }
  var _emwgpuDeviceDestroy = (devicePtr) => {
    const device = WebGPU.getJsObject(devicePtr);
    device.onuncapturederror = null;
    device.destroy();
  };
  function _emwgpuInstanceRequestAdapter(instancePtr, futureId, options, adapterPtr) {
    instancePtr >>>= 0;
    futureId = bigintToI53Checked(futureId);
    options >>>= 0;
    adapterPtr >>>= 0;
    var opts;
    if (options) {
      opts = { featureLevel: WebGPU.FeatureLevel[HEAP32[options + 4 >>> 2 >>> 0]], powerPreference: WebGPU.PowerPreference[HEAP32[options + 8 >>> 2 >>> 0]], forceFallbackAdapter: !!HEAPU32[options + 12 >>> 2 >>> 0] };
      var nextInChainPtr = HEAPU32[options >>> 2 >>> 0];
      if (nextInChainPtr !== 0) {
        var sType = HEAP32[nextInChainPtr + 4 >>> 2 >>> 0];
        var webxrOptions = nextInChainPtr;
        opts.xrCompatible = !!HEAPU32[webxrOptions + 8 >>> 2 >>> 0];
      }
    }
    if (!("gpu" in navigator)) {
      var sp = stackSave();
      var messagePtr = stringToUTF8OnStack("WebGPU not available on this browser (navigator.gpu is not available)");
      _emwgpuOnRequestAdapterCompleted(futureId, 3, adapterPtr, messagePtr);
      stackRestore(sp);
      return;
    }
    WebGPU.Internals.futureInsert(futureId, navigator.gpu.requestAdapter(opts).then((adapter) => {
      callUserCallback(() => {
        if (adapter) {
          WebGPU.Internals.jsObjectInsert(adapterPtr, adapter);
          _emwgpuOnRequestAdapterCompleted(futureId, 1, adapterPtr, 0);
        } else {
          var sp2 = stackSave();
          var messagePtr2 = stringToUTF8OnStack("WebGPU not available on this browser (requestAdapter returned null)");
          _emwgpuOnRequestAdapterCompleted(futureId, 3, adapterPtr, messagePtr2);
          stackRestore(sp2);
        }
      });
    }, (ex) => {
      callUserCallback(() => {
        var sp2 = stackSave();
        var messagePtr2 = stringToUTF8OnStack(ex.message);
        _emwgpuOnRequestAdapterCompleted(futureId, 4, adapterPtr, messagePtr2);
        stackRestore(sp2);
      });
    }));
  }
  var _emwgpuQueueOnSubmittedWorkDone = function(queuePtr, futureId) {
    queuePtr >>>= 0;
    futureId = bigintToI53Checked(futureId);
    var queue = WebGPU.getJsObject(queuePtr);
    WebGPU.Internals.futureInsert(futureId, queue.onSubmittedWorkDone().then(() => {
      callUserCallback(() => {
        _emwgpuOnWorkDoneCompleted(futureId, 1);
      });
    }));
  };
  var _emwgpuWaitAny = function(futurePtr, futureCount, timeoutMSPtr) {
    futurePtr >>>= 0;
    futureCount >>>= 0;
    timeoutMSPtr >>>= 0;
    return Asyncify.handleAsync(async () => {
      var promises = [];
      if (timeoutMSPtr) {
        var timeoutMS = HEAP32[timeoutMSPtr >>> 2 >>> 0];
        promises.length = futureCount + 1;
        promises[futureCount] = new Promise((resolve) => setTimeout(resolve, timeoutMS, 0));
      } else {
        promises.length = futureCount;
      }
      for (var i = 0;i < futureCount; ++i) {
        var futureId = readI53FromI64(futurePtr + i * 8);
        if (!(futureId in WebGPU.Internals.futures)) {
          return futureId;
        }
        promises[i] = WebGPU.Internals.futures[futureId];
      }
      const firstResolvedFuture = await Promise.race(promises);
      delete WebGPU.Internals.futures[firstResolvedFuture];
      return firstResolvedFuture;
    });
  };
  _emwgpuWaitAny.isAsync = true;
  var ENV = {};
  var getExecutableName = () => thisProgram || "./this.program";
  var getEnvStrings = () => {
    if (!getEnvStrings.strings) {
      var lang = (globalThis.navigator?.language ?? "C").replace("-", "_") + ".UTF-8";
      var env = { USER: "web_user", LOGNAME: "web_user", PATH: "/", PWD: "/", HOME: "/home/web_user", LANG: lang, _: getExecutableName() };
      for (var x in ENV) {
        if (ENV[x] === undefined)
          delete env[x];
        else
          env[x] = ENV[x];
      }
      var strings = [];
      for (var x in env) {
        strings.push(`${x}=${env[x]}`);
      }
      getEnvStrings.strings = strings;
    }
    return getEnvStrings.strings;
  };
  function _environ_get(__environ, environ_buf) {
    __environ >>>= 0;
    environ_buf >>>= 0;
    var bufSize = 0;
    var envp = 0;
    for (var string of getEnvStrings()) {
      var ptr = environ_buf + bufSize;
      HEAPU32[__environ + envp >>> 2 >>> 0] = ptr;
      bufSize += stringToUTF8(string, ptr, Infinity) + 1;
      envp += 4;
    }
    return 0;
  }
  function _environ_sizes_get(penviron_count, penviron_buf_size) {
    penviron_count >>>= 0;
    penviron_buf_size >>>= 0;
    var strings = getEnvStrings();
    HEAPU32[penviron_count >>> 2 >>> 0] = strings.length;
    var bufSize = 0;
    for (var string of strings) {
      bufSize += lengthBytesUTF8(string) + 1;
    }
    HEAPU32[penviron_buf_size >>> 2 >>> 0] = bufSize;
    return 0;
  }
  function _fd_close(fd) {
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      FS.close(stream);
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return e.errno;
    }
  }
  var doReadv = (stream, iov, iovcnt, offset) => {
    var ret = 0;
    for (var i = 0;i < iovcnt; i++) {
      var ptr = HEAPU32[iov >>> 2 >>> 0];
      var len = HEAPU32[iov + 4 >>> 2 >>> 0];
      iov += 8;
      var curr = FS.read(stream, HEAP8, ptr, len, offset);
      if (curr < 0)
        return -1;
      ret += curr;
      if (curr < len)
        break;
      if (typeof offset != "undefined") {
        offset += curr;
      }
    }
    return ret;
  };
  function _fd_read(fd, iov, iovcnt, pnum) {
    iov >>>= 0;
    iovcnt >>>= 0;
    pnum >>>= 0;
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      var num = doReadv(stream, iov, iovcnt);
      HEAPU32[pnum >>> 2 >>> 0] = num;
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return e.errno;
    }
  }
  function _fd_seek(fd, offset, whence, newOffset) {
    offset = bigintToI53Checked(offset);
    newOffset >>>= 0;
    try {
      if (isNaN(offset))
        return 22;
      var stream = SYSCALLS.getStreamFromFD(fd);
      FS.llseek(stream, offset, whence);
      HEAP64[newOffset >>> 3 >>> 0] = BigInt(stream.position);
      if (stream.getdents && offset === 0 && whence === 0)
        stream.getdents = null;
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return e.errno;
    }
  }
  var doWritev = (stream, iov, iovcnt, offset) => {
    var ret = 0;
    for (var i = 0;i < iovcnt; i++) {
      var ptr = HEAPU32[iov >>> 2 >>> 0];
      var len = HEAPU32[iov + 4 >>> 2 >>> 0];
      iov += 8;
      var curr = FS.write(stream, HEAP8, ptr, len, offset);
      if (curr < 0)
        return -1;
      ret += curr;
      if (curr < len) {
        break;
      }
      if (typeof offset != "undefined") {
        offset += curr;
      }
    }
    return ret;
  };
  function _fd_write(fd, iov, iovcnt, pnum) {
    iov >>>= 0;
    iovcnt >>>= 0;
    pnum >>>= 0;
    try {
      var stream = SYSCALLS.getStreamFromFD(fd);
      var num = doWritev(stream, iov, iovcnt);
      HEAPU32[pnum >>> 2 >>> 0] = num;
      return 0;
    } catch (e) {
      if (typeof FS == "undefined" || !(e.name === "ErrnoError"))
        throw e;
      return e.errno;
    }
  }
  var emwgpuStringToInt_FeatureName = { "core-features-and-limits": 1, "depth-clip-control": 2, "depth32float-stencil8": 3, "texture-compression-bc": 4, "texture-compression-bc-sliced-3d": 5, "texture-compression-etc2": 6, "texture-compression-astc": 7, "texture-compression-astc-sliced-3d": 8, "timestamp-query": 9, "indirect-first-instance": 10, "shader-f16": 11, "rg11b10ufloat-renderable": 12, "bgra8unorm-storage": 13, "float32-filterable": 14, "float32-blendable": 15, "clip-distances": 16, "dual-source-blending": 17, subgroups: 18, "texture-formats-tier1": 19, "texture-formats-tier2": 20, "primitive-index": 21, "texture-component-swizzle": 22, "chromium-experimental-unorm16-texture-formats": 327692, "chromium-experimental-multi-draw-indirect": 327729 };
  function _wgpuAdapterGetFeatures(adapterPtr, supportedFeatures) {
    adapterPtr >>>= 0;
    supportedFeatures >>>= 0;
    var adapter = WebGPU.getJsObject(adapterPtr);
    var featuresPtr = _malloc(adapter.features.size * 4);
    var offset = 0;
    var numFeatures = 0;
    for (const feature of adapter.features) {
      var featureEnumValue = emwgpuStringToInt_FeatureName[feature];
      if (featureEnumValue >= 0) {
        HEAP32[featuresPtr + offset >>> 2 >>> 0] = featureEnumValue;
        offset += 4;
        numFeatures++;
      }
    }
    HEAPU32[supportedFeatures + 4 >>> 2 >>> 0] = featuresPtr;
    HEAPU32[supportedFeatures >>> 2 >>> 0] = numFeatures;
  }
  function _wgpuAdapterGetInfo(adapterPtr, info) {
    adapterPtr >>>= 0;
    info >>>= 0;
    var adapter = WebGPU.getJsObject(adapterPtr);
    WebGPU.fillAdapterInfoStruct(adapter.info, info);
    return 1;
  }
  function _wgpuAdapterGetLimits(adapterPtr, limitsOutPtr) {
    adapterPtr >>>= 0;
    limitsOutPtr >>>= 0;
    var adapter = WebGPU.getJsObject(adapterPtr);
    WebGPU.fillLimitStruct(adapter.limits, limitsOutPtr);
    return 1;
  }
  function _wgpuAdapterHasFeature(adapterPtr, featureEnumValue) {
    adapterPtr >>>= 0;
    var adapter = WebGPU.getJsObject(adapterPtr);
    return adapter.features.has(WebGPU.FeatureName[featureEnumValue]);
  }
  var _wgpuBufferGetSize = function(bufferPtr) {
    bufferPtr >>>= 0;
    var ret = (() => {
      var buffer = WebGPU.getJsObject(bufferPtr);
      return buffer.size;
    })();
    return BigInt(ret);
  };
  function _wgpuCommandEncoderBeginComputePass(encoderPtr, descriptor) {
    encoderPtr >>>= 0;
    descriptor >>>= 0;
    var desc;
    if (descriptor) {
      desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4), timestampWrites: WebGPU.makePassTimestampWrites(HEAPU32[descriptor + 12 >>> 2 >>> 0]) };
    }
    var commandEncoder = WebGPU.getJsObject(encoderPtr);
    var ptr = _emwgpuCreateComputePassEncoder(0);
    WebGPU.Internals.jsObjectInsert(ptr, commandEncoder.beginComputePass(desc));
    return ptr;
  }
  function _wgpuCommandEncoderCopyBufferToBuffer(encoderPtr, srcPtr, srcOffset, dstPtr, dstOffset, size) {
    encoderPtr >>>= 0;
    srcPtr >>>= 0;
    srcOffset = bigintToI53Checked(srcOffset);
    dstPtr >>>= 0;
    dstOffset = bigintToI53Checked(dstOffset);
    size = bigintToI53Checked(size);
    var commandEncoder = WebGPU.getJsObject(encoderPtr);
    var src = WebGPU.getJsObject(srcPtr);
    var dst = WebGPU.getJsObject(dstPtr);
    commandEncoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
  }
  function _wgpuCommandEncoderFinish(encoderPtr, descriptor) {
    encoderPtr >>>= 0;
    descriptor >>>= 0;
    var commandEncoder = WebGPU.getJsObject(encoderPtr);
    var ptr = _emwgpuCreateCommandBuffer(0);
    WebGPU.Internals.jsObjectInsert(ptr, commandEncoder.finish());
    return ptr;
  }
  function _wgpuCommandEncoderResolveQuerySet(encoderPtr, querySetPtr, firstQuery, queryCount, destinationPtr, destinationOffset) {
    encoderPtr >>>= 0;
    querySetPtr >>>= 0;
    destinationPtr >>>= 0;
    destinationOffset = bigintToI53Checked(destinationOffset);
    var commandEncoder = WebGPU.getJsObject(encoderPtr);
    var querySet = WebGPU.getJsObject(querySetPtr);
    var destination = WebGPU.getJsObject(destinationPtr);
    commandEncoder.resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset);
  }
  function _wgpuComputePassEncoderDispatchWorkgroups(passPtr, x, y, z) {
    passPtr >>>= 0;
    var pass = WebGPU.getJsObject(passPtr);
    pass.dispatchWorkgroups(x, y, z);
  }
  function _wgpuComputePassEncoderEnd(passPtr) {
    passPtr >>>= 0;
    var pass = WebGPU.getJsObject(passPtr);
    pass.end();
  }
  function _wgpuComputePassEncoderSetBindGroup(passPtr, groupIndex, groupPtr, dynamicOffsetCount, dynamicOffsetsPtr) {
    passPtr >>>= 0;
    groupPtr >>>= 0;
    dynamicOffsetCount >>>= 0;
    dynamicOffsetsPtr >>>= 0;
    var pass = WebGPU.getJsObject(passPtr);
    var group = WebGPU.getJsObject(groupPtr);
    if (dynamicOffsetCount == 0) {
      pass.setBindGroup(groupIndex, group);
    } else {
      pass.setBindGroup(groupIndex, group, HEAPU32, dynamicOffsetsPtr >>> 2, dynamicOffsetCount);
    }
  }
  function _wgpuComputePassEncoderSetPipeline(passPtr, pipelinePtr) {
    passPtr >>>= 0;
    pipelinePtr >>>= 0;
    var pass = WebGPU.getJsObject(passPtr);
    var pipeline = WebGPU.getJsObject(pipelinePtr);
    pass.setPipeline(pipeline);
  }
  function _wgpuComputePipelineGetBindGroupLayout(pipelinePtr, groupIndex) {
    pipelinePtr >>>= 0;
    var pipeline = WebGPU.getJsObject(pipelinePtr);
    var ptr = _emwgpuCreateBindGroupLayout(0);
    WebGPU.Internals.jsObjectInsert(ptr, pipeline.getBindGroupLayout(groupIndex));
    return ptr;
  }
  var _wgpuDeviceCreateBindGroup = function(devicePtr, descriptor) {
    devicePtr >>>= 0;
    descriptor >>>= 0;
    function makeEntry(entryPtr) {
      var bufferPtr = HEAPU32[entryPtr + 8 >>> 2 >>> 0];
      var samplerPtr = HEAPU32[entryPtr + 32 >>> 2 >>> 0];
      var textureViewPtr = HEAPU32[entryPtr + 36 >>> 2 >>> 0];
      var externalTexturePtr = 0;
      WebGPU.iterateExtensions(entryPtr, { 14: (ptr2) => {
        externalTexturePtr = HEAPU32[ptr2 + 8 >>> 2 >>> 0];
      } });
      var resource;
      if (bufferPtr) {
        var size = readI53FromI64(entryPtr + 24);
        if (size == -1)
          size = undefined;
        resource = { buffer: WebGPU.getJsObject(bufferPtr), offset: readI53FromI64(entryPtr + 16), size };
      } else {
        resource = WebGPU.getJsObject(samplerPtr || textureViewPtr || externalTexturePtr);
      }
      return { binding: HEAPU32[entryPtr + 4 >>> 2 >>> 0], resource };
    }
    function makeEntries(count, entriesPtrs) {
      var entries = [];
      for (var i = 0;i < count; ++i) {
        entries.push(makeEntry(entriesPtrs + 40 * i));
      }
      return entries;
    }
    var desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4), layout: WebGPU.getJsObject(HEAPU32[descriptor + 12 >>> 2 >>> 0]), entries: makeEntries(HEAPU32[descriptor + 16 >>> 2 >>> 0], HEAPU32[descriptor + 20 >>> 2 >>> 0]) };
    var device = WebGPU.getJsObject(devicePtr);
    var ptr = _emwgpuCreateBindGroup(0);
    WebGPU.Internals.jsObjectInsert(ptr, device.createBindGroup(desc));
    return ptr;
  };
  function _wgpuDeviceCreateCommandEncoder(devicePtr, descriptor) {
    devicePtr >>>= 0;
    descriptor >>>= 0;
    var desc;
    if (descriptor) {
      desc = { label: WebGPU.makeStringFromOptionalStringView(descriptor + 4) };
    }
    var device = WebGPU.getJsObject(devicePtr);
    var ptr = _emwgpuCreateCommandEncoder(0);
    WebGPU.Internals.jsObjectInsert(ptr, device.createCommandEncoder(desc));
    return ptr;
  }
  function _wgpuDeviceCreateComputePipeline(devicePtr, descriptor) {
    devicePtr >>>= 0;
    descriptor >>>= 0;
    var desc = WebGPU.makeComputePipelineDesc(descriptor);
    var device = WebGPU.getJsObject(devicePtr);
    var ptr = _emwgpuCreateComputePipeline(0);
    WebGPU.Internals.jsObjectInsert(ptr, device.createComputePipeline(desc));
    return ptr;
  }
  function _wgpuDeviceCreateQuerySet(devicePtr, descriptor) {
    devicePtr >>>= 0;
    descriptor >>>= 0;
    var desc = { type: WebGPU.QueryType[HEAP32[descriptor + 12 >>> 2 >>> 0]], count: HEAPU32[descriptor + 16 >>> 2 >>> 0] };
    var device = WebGPU.getJsObject(devicePtr);
    var ptr = _emwgpuCreateQuerySet(0);
    WebGPU.Internals.jsObjectInsert(ptr, device.createQuerySet(desc));
    return ptr;
  }
  function _wgpuQuerySetDestroy(querySetPtr) {
    querySetPtr >>>= 0;
    WebGPU.getJsObject(querySetPtr).destroy();
  }
  var _wgpuQueueSubmit = function(queuePtr, commandCount, commands) {
    queuePtr >>>= 0;
    commandCount >>>= 0;
    commands >>>= 0;
    var queue = WebGPU.getJsObject(queuePtr);
    var cmds = Array.from(HEAP32.subarray(commands >>> 2 >>> 0, commands + commandCount * 4 >>> 2 >>> 0), (id) => WebGPU.getJsObject(id));
    queue.submit(cmds);
  };
  function _wgpuQueueWriteBuffer(queuePtr, bufferPtr, bufferOffset, data, size) {
    queuePtr >>>= 0;
    bufferPtr >>>= 0;
    bufferOffset = bigintToI53Checked(bufferOffset);
    data >>>= 0;
    size >>>= 0;
    var queue = WebGPU.getJsObject(queuePtr);
    var buffer = WebGPU.getJsObject(bufferPtr);
    var subarray = HEAPU8.subarray(data >>> 0, data + size >>> 0);
    queue.writeBuffer(buffer, bufferOffset, subarray, 0, size);
  }
  var Asyncify = { instrumentWasmImports(imports) {
    var importPattern = /^(invoke_.*|__asyncjs__.*)$/;
    for (let [x, original] of Object.entries(imports)) {
      if (typeof original == "function") {
        let isAsyncifyImport = original.isAsync || importPattern.test(x);
        if (isAsyncifyImport) {
          imports[x] = original = new WebAssembly.Suspending(original);
        }
      }
    }
  }, instrumentWasmExports(exports) {
    var exportPattern = /^(webgpu_init|ctx_create|ctx_free|backend_alloc_ctx_tensors|backend_buffer_free|backend_tensor_set|backend_tensor_set3|backend_tensor_get|backend_tensor_get_async_begin|backend_tensor_get_async_poll|backend_tensor_get_async_finish|backend_tensor_get_async_cancel|graph_compute|webllm_load_model|webllm_free_model|webllm_create_context|webllm_free_context|webllm_decode|webllm_get_logits|webllm_get_embeddings|main|__main_argc_argv)$/;
    Asyncify.asyncExports = new Set;
    var ret = {};
    for (let [x, original] of Object.entries(exports)) {
      if (typeof original == "function") {
        let isAsyncifyExport = exportPattern.test(x);
        if (isAsyncifyExport) {
          Asyncify.asyncExports.add(original);
          original = Asyncify.makeAsyncFunction(original);
        }
        ret[x] = original;
      } else {
        ret[x] = original;
      }
    }
    return ret;
  }, asyncExports: null, isAsyncExport(func) {
    return Asyncify.asyncExports?.has(func);
  }, handleAsync: async (startAsync) => {
    try {
      return await startAsync();
    } finally {}
  }, handleSleep: (startAsync) => Asyncify.handleAsync(() => new Promise(startAsync)), makeAsyncFunction(original) {
    return WebAssembly.promising(original);
  } };
  var getCFunc = (ident) => {
    var func = Module2["_" + ident];
    return func;
  };
  var writeArrayToMemory = (array, buffer) => {
    HEAP8.set(array, buffer >>> 0);
  };
  var ccall = (ident, returnType, argTypes, args, opts) => {
    var toC = { string: (str) => {
      var ret2 = 0;
      if (str !== null && str !== undefined && str !== 0) {
        ret2 = stringToUTF8OnStack(str);
      }
      return ret2;
    }, array: (arr) => {
      var ret2 = stackAlloc(arr.length);
      writeArrayToMemory(arr, ret2);
      return ret2;
    } };
    function convertReturnValue(ret2) {
      if (returnType === "string") {
        return UTF8ToString(ret2);
      }
      if (returnType === "pointer")
        return ret2 >>> 0;
      if (returnType === "boolean")
        return Boolean(ret2);
      return ret2;
    }
    var func = getCFunc(ident);
    var cArgs = [];
    var stack = 0;
    if (args) {
      for (var i = 0;i < args.length; i++) {
        var converter = toC[argTypes[i]];
        if (converter) {
          if (stack === 0)
            stack = stackSave();
          cArgs[i] = converter(args[i]);
        } else {
          cArgs[i] = args[i];
        }
      }
    }
    var ret = func(...cArgs);
    function onDone(ret2) {
      if (stack !== 0)
        stackRestore(stack);
      return convertReturnValue(ret2);
    }
    var asyncMode = opts?.async;
    if (asyncMode)
      return ret.then(onDone);
    ret = onDone(ret);
    return ret;
  };
  var cwrap = (ident, returnType, argTypes, opts) => {
    var numericArgs = !argTypes || argTypes.every((type) => type === "number" || type === "boolean");
    var numericRet = returnType !== "string";
    if (numericRet && numericArgs && !opts) {
      return getCFunc(ident);
    }
    return (...args) => ccall(ident, returnType, argTypes, args, opts);
  };
  FS.createPreloadedFile = FS_createPreloadedFile;
  FS.preloadFile = FS_preloadFile;
  FS.staticInit();
  {
    if (Module2["noExitRuntime"])
      noExitRuntime = Module2["noExitRuntime"];
    if (Module2["preloadPlugins"])
      preloadPlugins = Module2["preloadPlugins"];
    if (Module2["print"])
      out = Module2["print"];
    if (Module2["printErr"])
      err = Module2["printErr"];
    if (Module2["wasmBinary"])
      wasmBinary = Module2["wasmBinary"];
    if (Module2["arguments"])
      arguments_ = Module2["arguments"];
    if (Module2["thisProgram"])
      thisProgram = Module2["thisProgram"];
    if (Module2["preInit"]) {
      if (typeof Module2["preInit"] == "function")
        Module2["preInit"] = [Module2["preInit"]];
      while (Module2["preInit"].length > 0) {
        Module2["preInit"].shift()();
      }
    }
  }
  Module2["stackSave"] = stackSave;
  Module2["stackRestore"] = stackRestore;
  Module2["stackAlloc"] = stackAlloc;
  Module2["cwrap"] = cwrap;
  Module2["stringToUTF8"] = stringToUTF8;
  Module2["lengthBytesUTF8"] = lengthBytesUTF8;
  Module2["Asyncify"] = Asyncify;
  var ASM_CONSTS = { 1478113: ($0, $1, $2, $3) => Module2.jsepRunOp ? Module2.jsepRunOp($0, $1, $2, $3) : 1 };
  function ggml_webgpu_notify_async_tensor_get(request_id, state) {
    Module2.__webllmNotifyAsyncTensorGet?.(request_id, state);
  }
  ggml_webgpu_notify_async_tensor_get.sig = "vii";
  function ggml_jsep_alloc(size) {
    if (Module2.jsepAlloc) {
      return Module2.jsepAlloc(size) | 0;
    }
    Module2.__jsep_stub_handle = (Module2.__jsep_stub_handle | 0) + 1;
    return Module2.__jsep_stub_handle;
  }
  ggml_jsep_alloc.sig = "ii";
  function ggml_jsep_free(handle) {
    if (Module2.jsepFree) {
      Module2.jsepFree(handle);
    }
  }
  ggml_jsep_free.sig = "vi";
  function ggml_jsep_write(handle, offset, host_ptr, size) {
    if (Module2.jsepWrite) {
      Module2.jsepWrite(handle, offset, host_ptr, size);
    }
  }
  ggml_jsep_write.sig = "viiii";
  function ggml_jsep_read(handle, offset, host_ptr, size) {
    if (Module2.jsepRead) {
      return Module2.jsepRead(handle, offset, host_ptr, size);
    }
  }
  ggml_jsep_read.sig = "viiii";
  function ggml_jsep_clear(handle, value, offset, size) {
    if (Module2.jsepClear) {
      Module2.jsepClear(handle, value, offset, size);
    }
  }
  ggml_jsep_clear.sig = "viiii";
  function ggml_jsep_sync() {
    if (Module2.jsepSync) {
      Module2.jsepSync();
    }
  }
  ggml_jsep_sync.sig = "v";
  var _bridge_malloc, _malloc, _bridge_free, _free, _webgpu_init, _webgpu_shutdown, _ctx_create, _ctx_free, _tensor_new_1d, _tensor_new_2d, _tensor_new_3d, _tensor_new_4d, _tensor_set_name, _tensor_nelements, _tensor_nbytes, _tensor_type, _tensor_ne, _tensor_nb, _tensor_data, _tensor_set_data, _tensor_get_data, _op_mul_mat, _op_add, _op_mul, _op_rms_norm, _op_silu, _op_gelu, _op_rope, _op_reshape_2d, _op_reshape_3d, _op_permute, _op_cont, _op_view_2d, _op_view_3d, _op_cpy, _op_soft_max, _op_soft_max_ext, _op_flash_attn_ext, _op_flash_attn_ext_set_prec, _op_flash_attn_ext_add_sinks, _op_swiglu_split, _op_scale, _op_repeat, _op_get_rows, _op_argmax, _op_top_k, _op_diag_mask_inf, _op_norm, _graph_new, _graph_build_forward_expand, _graph_compute, _backend_alloc_ctx_tensors, _backend_buffer_free, _backend_tensor_set, _backend_tensor_set3, _backend_tensor_get, _backend_tensor_get_async_begin, _backend_tensor_get_async_poll, _backend_tensor_get_async_finish, _backend_tensor_get_async_cancel, _backend_tensor_get_async_callback_support, _backend_tensor_alignment, _webgpu_set_graph_profiling_enabled, _webgpu_last_graph_profile_valid, _webgpu_last_graph_profile_breakdown_available, _webgpu_last_graph_profile_total_ms, _webgpu_last_graph_profile_matmul_ms, _webgpu_last_graph_profile_attention_ms, _webgpu_last_graph_profile_encode_overhead_ms, _webgpu_last_graph_profile_dispatch_count, _webllm_load_model, _webllm_free_model, _webllm_create_context, _webllm_free_context, _webllm_decode, _webllm_get_logits, _webllm_n_vocab, _webllm_tokenize, _webllm_detokenize, _webllm_token_bos, _webllm_token_eos, _webllm_get_metadata, _webllm_n_ctx_train, _webllm_n_embd, _webllm_n_layer, _webllm_n_head, _webllm_n_head_kv, _webllm_n_ctx, _webllm_kv_seq_rm, _webllm_kv_clear, _webllm_state_seq_get_size, _webllm_state_seq_get_data, _webllm_state_seq_set_data, _webllm_get_embeddings, _webllm_perf_counter, _emwgpuCreateBindGroup, _emwgpuCreateBindGroupLayout, _emwgpuCreateCommandBuffer, _emwgpuCreateCommandEncoder, _emwgpuCreateComputePassEncoder, _emwgpuCreateComputePipeline, _emwgpuCreateExternalTexture, _emwgpuCreatePipelineLayout, _emwgpuCreateQuerySet, _emwgpuCreateRenderBundle, _emwgpuCreateRenderBundleEncoder, _emwgpuCreateRenderPassEncoder, _emwgpuCreateRenderPipeline, _emwgpuCreateSampler, _emwgpuCreateSurface, _emwgpuCreateTexture, _emwgpuCreateTextureView, _emwgpuCreateAdapter, _emwgpuImportBuffer, _emwgpuCreateDevice, _emwgpuCreateQueue, _emwgpuCreateShaderModule, _emwgpuOnDeviceLostCompleted, _emwgpuOnMapAsyncCompleted, _emwgpuOnRequestAdapterCompleted, _emwgpuOnRequestDeviceCompleted, _emwgpuOnWorkDoneCompleted, _emwgpuOnUncapturedError, _emscripten_builtin_memalign, _memalign, ___trap, __emscripten_stack_restore, __emscripten_stack_alloc, _emscripten_stack_get_current, memory, __indirect_function_table, wasmMemory, wasmTable;
  function assignWasmExports(wasmExports2) {
    _bridge_malloc = Module2["_bridge_malloc"] = wasmExports2["bridge_malloc"];
    _malloc = Module2["_malloc"] = wasmExports2["malloc"];
    _bridge_free = Module2["_bridge_free"] = wasmExports2["bridge_free"];
    _free = Module2["_free"] = wasmExports2["free"];
    _webgpu_init = Module2["_webgpu_init"] = wasmExports2["webgpu_init"];
    _webgpu_shutdown = Module2["_webgpu_shutdown"] = wasmExports2["webgpu_shutdown"];
    _ctx_create = Module2["_ctx_create"] = wasmExports2["ctx_create"];
    _ctx_free = Module2["_ctx_free"] = wasmExports2["ctx_free"];
    _tensor_new_1d = Module2["_tensor_new_1d"] = wasmExports2["tensor_new_1d"];
    _tensor_new_2d = Module2["_tensor_new_2d"] = wasmExports2["tensor_new_2d"];
    _tensor_new_3d = Module2["_tensor_new_3d"] = wasmExports2["tensor_new_3d"];
    _tensor_new_4d = Module2["_tensor_new_4d"] = wasmExports2["tensor_new_4d"];
    _tensor_set_name = Module2["_tensor_set_name"] = wasmExports2["tensor_set_name"];
    _tensor_nelements = Module2["_tensor_nelements"] = wasmExports2["tensor_nelements"];
    _tensor_nbytes = Module2["_tensor_nbytes"] = wasmExports2["tensor_nbytes"];
    _tensor_type = Module2["_tensor_type"] = wasmExports2["tensor_type"];
    _tensor_ne = Module2["_tensor_ne"] = wasmExports2["tensor_ne"];
    _tensor_nb = Module2["_tensor_nb"] = wasmExports2["tensor_nb"];
    _tensor_data = Module2["_tensor_data"] = wasmExports2["tensor_data"];
    _tensor_set_data = Module2["_tensor_set_data"] = wasmExports2["tensor_set_data"];
    _tensor_get_data = Module2["_tensor_get_data"] = wasmExports2["tensor_get_data"];
    _op_mul_mat = Module2["_op_mul_mat"] = wasmExports2["op_mul_mat"];
    _op_add = Module2["_op_add"] = wasmExports2["op_add"];
    _op_mul = Module2["_op_mul"] = wasmExports2["op_mul"];
    _op_rms_norm = Module2["_op_rms_norm"] = wasmExports2["op_rms_norm"];
    _op_silu = Module2["_op_silu"] = wasmExports2["op_silu"];
    _op_gelu = Module2["_op_gelu"] = wasmExports2["op_gelu"];
    _op_rope = Module2["_op_rope"] = wasmExports2["op_rope"];
    _op_reshape_2d = Module2["_op_reshape_2d"] = wasmExports2["op_reshape_2d"];
    _op_reshape_3d = Module2["_op_reshape_3d"] = wasmExports2["op_reshape_3d"];
    _op_permute = Module2["_op_permute"] = wasmExports2["op_permute"];
    _op_cont = Module2["_op_cont"] = wasmExports2["op_cont"];
    _op_view_2d = Module2["_op_view_2d"] = wasmExports2["op_view_2d"];
    _op_view_3d = Module2["_op_view_3d"] = wasmExports2["op_view_3d"];
    _op_cpy = Module2["_op_cpy"] = wasmExports2["op_cpy"];
    _op_soft_max = Module2["_op_soft_max"] = wasmExports2["op_soft_max"];
    _op_soft_max_ext = Module2["_op_soft_max_ext"] = wasmExports2["op_soft_max_ext"];
    _op_flash_attn_ext = Module2["_op_flash_attn_ext"] = wasmExports2["op_flash_attn_ext"];
    _op_flash_attn_ext_set_prec = Module2["_op_flash_attn_ext_set_prec"] = wasmExports2["op_flash_attn_ext_set_prec"];
    _op_flash_attn_ext_add_sinks = Module2["_op_flash_attn_ext_add_sinks"] = wasmExports2["op_flash_attn_ext_add_sinks"];
    _op_swiglu_split = Module2["_op_swiglu_split"] = wasmExports2["op_swiglu_split"];
    _op_scale = Module2["_op_scale"] = wasmExports2["op_scale"];
    _op_repeat = Module2["_op_repeat"] = wasmExports2["op_repeat"];
    _op_get_rows = Module2["_op_get_rows"] = wasmExports2["op_get_rows"];
    _op_argmax = Module2["_op_argmax"] = wasmExports2["op_argmax"];
    _op_top_k = Module2["_op_top_k"] = wasmExports2["op_top_k"];
    _op_diag_mask_inf = Module2["_op_diag_mask_inf"] = wasmExports2["op_diag_mask_inf"];
    _op_norm = Module2["_op_norm"] = wasmExports2["op_norm"];
    _graph_new = Module2["_graph_new"] = wasmExports2["graph_new"];
    _graph_build_forward_expand = Module2["_graph_build_forward_expand"] = wasmExports2["graph_build_forward_expand"];
    _graph_compute = Module2["_graph_compute"] = wasmExports2["graph_compute"];
    _backend_alloc_ctx_tensors = Module2["_backend_alloc_ctx_tensors"] = wasmExports2["backend_alloc_ctx_tensors"];
    _backend_buffer_free = Module2["_backend_buffer_free"] = wasmExports2["backend_buffer_free"];
    _backend_tensor_set = Module2["_backend_tensor_set"] = wasmExports2["backend_tensor_set"];
    _backend_tensor_set3 = Module2["_backend_tensor_set3"] = wasmExports2["backend_tensor_set3"];
    _backend_tensor_get = Module2["_backend_tensor_get"] = wasmExports2["backend_tensor_get"];
    _backend_tensor_get_async_begin = Module2["_backend_tensor_get_async_begin"] = wasmExports2["backend_tensor_get_async_begin"];
    _backend_tensor_get_async_poll = Module2["_backend_tensor_get_async_poll"] = wasmExports2["backend_tensor_get_async_poll"];
    _backend_tensor_get_async_finish = Module2["_backend_tensor_get_async_finish"] = wasmExports2["backend_tensor_get_async_finish"];
    _backend_tensor_get_async_cancel = Module2["_backend_tensor_get_async_cancel"] = wasmExports2["backend_tensor_get_async_cancel"];
    _backend_tensor_get_async_callback_support = Module2["_backend_tensor_get_async_callback_support"] = wasmExports2["backend_tensor_get_async_callback_support"];
    _backend_tensor_alignment = Module2["_backend_tensor_alignment"] = wasmExports2["backend_tensor_alignment"];
    _webgpu_set_graph_profiling_enabled = Module2["_webgpu_set_graph_profiling_enabled"] = wasmExports2["webgpu_set_graph_profiling_enabled"];
    _webgpu_last_graph_profile_valid = Module2["_webgpu_last_graph_profile_valid"] = wasmExports2["webgpu_last_graph_profile_valid"];
    _webgpu_last_graph_profile_breakdown_available = Module2["_webgpu_last_graph_profile_breakdown_available"] = wasmExports2["webgpu_last_graph_profile_breakdown_available"];
    _webgpu_last_graph_profile_total_ms = Module2["_webgpu_last_graph_profile_total_ms"] = wasmExports2["webgpu_last_graph_profile_total_ms"];
    _webgpu_last_graph_profile_matmul_ms = Module2["_webgpu_last_graph_profile_matmul_ms"] = wasmExports2["webgpu_last_graph_profile_matmul_ms"];
    _webgpu_last_graph_profile_attention_ms = Module2["_webgpu_last_graph_profile_attention_ms"] = wasmExports2["webgpu_last_graph_profile_attention_ms"];
    _webgpu_last_graph_profile_encode_overhead_ms = Module2["_webgpu_last_graph_profile_encode_overhead_ms"] = wasmExports2["webgpu_last_graph_profile_encode_overhead_ms"];
    _webgpu_last_graph_profile_dispatch_count = Module2["_webgpu_last_graph_profile_dispatch_count"] = wasmExports2["webgpu_last_graph_profile_dispatch_count"];
    _webllm_load_model = Module2["_webllm_load_model"] = wasmExports2["webllm_load_model"];
    _webllm_free_model = Module2["_webllm_free_model"] = wasmExports2["webllm_free_model"];
    _webllm_create_context = Module2["_webllm_create_context"] = wasmExports2["webllm_create_context"];
    _webllm_free_context = Module2["_webllm_free_context"] = wasmExports2["webllm_free_context"];
    _webllm_decode = Module2["_webllm_decode"] = wasmExports2["webllm_decode"];
    _webllm_get_logits = Module2["_webllm_get_logits"] = wasmExports2["webllm_get_logits"];
    _webllm_n_vocab = Module2["_webllm_n_vocab"] = wasmExports2["webllm_n_vocab"];
    _webllm_tokenize = Module2["_webllm_tokenize"] = wasmExports2["webllm_tokenize"];
    _webllm_detokenize = Module2["_webllm_detokenize"] = wasmExports2["webllm_detokenize"];
    _webllm_token_bos = Module2["_webllm_token_bos"] = wasmExports2["webllm_token_bos"];
    _webllm_token_eos = Module2["_webllm_token_eos"] = wasmExports2["webllm_token_eos"];
    _webllm_get_metadata = Module2["_webllm_get_metadata"] = wasmExports2["webllm_get_metadata"];
    _webllm_n_ctx_train = Module2["_webllm_n_ctx_train"] = wasmExports2["webllm_n_ctx_train"];
    _webllm_n_embd = Module2["_webllm_n_embd"] = wasmExports2["webllm_n_embd"];
    _webllm_n_layer = Module2["_webllm_n_layer"] = wasmExports2["webllm_n_layer"];
    _webllm_n_head = Module2["_webllm_n_head"] = wasmExports2["webllm_n_head"];
    _webllm_n_head_kv = Module2["_webllm_n_head_kv"] = wasmExports2["webllm_n_head_kv"];
    _webllm_n_ctx = Module2["_webllm_n_ctx"] = wasmExports2["webllm_n_ctx"];
    _webllm_kv_seq_rm = Module2["_webllm_kv_seq_rm"] = wasmExports2["webllm_kv_seq_rm"];
    _webllm_kv_clear = Module2["_webllm_kv_clear"] = wasmExports2["webllm_kv_clear"];
    _webllm_state_seq_get_size = Module2["_webllm_state_seq_get_size"] = wasmExports2["webllm_state_seq_get_size"];
    _webllm_state_seq_get_data = Module2["_webllm_state_seq_get_data"] = wasmExports2["webllm_state_seq_get_data"];
    _webllm_state_seq_set_data = Module2["_webllm_state_seq_set_data"] = wasmExports2["webllm_state_seq_set_data"];
    _webllm_get_embeddings = Module2["_webllm_get_embeddings"] = wasmExports2["webllm_get_embeddings"];
    _webllm_perf_counter = Module2["_webllm_perf_counter"] = wasmExports2["webllm_perf_counter"];
    _emwgpuCreateBindGroup = wasmExports2["emwgpuCreateBindGroup"];
    _emwgpuCreateBindGroupLayout = wasmExports2["emwgpuCreateBindGroupLayout"];
    _emwgpuCreateCommandBuffer = wasmExports2["emwgpuCreateCommandBuffer"];
    _emwgpuCreateCommandEncoder = wasmExports2["emwgpuCreateCommandEncoder"];
    _emwgpuCreateComputePassEncoder = wasmExports2["emwgpuCreateComputePassEncoder"];
    _emwgpuCreateComputePipeline = wasmExports2["emwgpuCreateComputePipeline"];
    _emwgpuCreateExternalTexture = wasmExports2["emwgpuCreateExternalTexture"];
    _emwgpuCreatePipelineLayout = wasmExports2["emwgpuCreatePipelineLayout"];
    _emwgpuCreateQuerySet = wasmExports2["emwgpuCreateQuerySet"];
    _emwgpuCreateRenderBundle = wasmExports2["emwgpuCreateRenderBundle"];
    _emwgpuCreateRenderBundleEncoder = wasmExports2["emwgpuCreateRenderBundleEncoder"];
    _emwgpuCreateRenderPassEncoder = wasmExports2["emwgpuCreateRenderPassEncoder"];
    _emwgpuCreateRenderPipeline = wasmExports2["emwgpuCreateRenderPipeline"];
    _emwgpuCreateSampler = wasmExports2["emwgpuCreateSampler"];
    _emwgpuCreateSurface = wasmExports2["emwgpuCreateSurface"];
    _emwgpuCreateTexture = wasmExports2["emwgpuCreateTexture"];
    _emwgpuCreateTextureView = wasmExports2["emwgpuCreateTextureView"];
    _emwgpuCreateAdapter = wasmExports2["emwgpuCreateAdapter"];
    _emwgpuImportBuffer = wasmExports2["emwgpuImportBuffer"];
    _emwgpuCreateDevice = wasmExports2["emwgpuCreateDevice"];
    _emwgpuCreateQueue = wasmExports2["emwgpuCreateQueue"];
    _emwgpuCreateShaderModule = wasmExports2["emwgpuCreateShaderModule"];
    _emwgpuOnDeviceLostCompleted = wasmExports2["emwgpuOnDeviceLostCompleted"];
    _emwgpuOnMapAsyncCompleted = wasmExports2["emwgpuOnMapAsyncCompleted"];
    _emwgpuOnRequestAdapterCompleted = wasmExports2["emwgpuOnRequestAdapterCompleted"];
    _emwgpuOnRequestDeviceCompleted = wasmExports2["emwgpuOnRequestDeviceCompleted"];
    _emwgpuOnWorkDoneCompleted = wasmExports2["emwgpuOnWorkDoneCompleted"];
    _emwgpuOnUncapturedError = wasmExports2["emwgpuOnUncapturedError"];
    _emscripten_builtin_memalign = wasmExports2["emscripten_builtin_memalign"];
    _memalign = wasmExports2["memalign"];
    ___trap = wasmExports2["__trap"];
    __emscripten_stack_restore = wasmExports2["_emscripten_stack_restore"];
    __emscripten_stack_alloc = wasmExports2["_emscripten_stack_alloc"];
    _emscripten_stack_get_current = wasmExports2["emscripten_stack_get_current"];
    memory = wasmMemory = wasmExports2["memory"];
    __indirect_function_table = wasmTable = wasmExports2["__indirect_function_table"];
  }
  var wasmImports = { __syscall_fcntl64: ___syscall_fcntl64, __syscall_ioctl: ___syscall_ioctl, __syscall_openat: ___syscall_openat, __syscall_rmdir: ___syscall_rmdir, __syscall_unlinkat: ___syscall_unlinkat, _abort_js: __abort_js, _mmap_js: __mmap_js, _munmap_js: __munmap_js, _tzset_js: __tzset_js, clock_time_get: _clock_time_get, emscripten_asm_const_int: _emscripten_asm_const_int, emscripten_get_heap_max: _emscripten_get_heap_max, emscripten_has_asyncify: _emscripten_has_asyncify, emscripten_resize_heap: _emscripten_resize_heap, emwgpuAdapterRequestDevice: _emwgpuAdapterRequestDevice, emwgpuBufferDestroy: _emwgpuBufferDestroy, emwgpuBufferGetConstMappedRange: _emwgpuBufferGetConstMappedRange, emwgpuBufferMapAsync: _emwgpuBufferMapAsync, emwgpuBufferUnmap: _emwgpuBufferUnmap, emwgpuDelete: _emwgpuDelete, emwgpuDeviceCreateBuffer: _emwgpuDeviceCreateBuffer, emwgpuDeviceCreateShaderModule: _emwgpuDeviceCreateShaderModule, emwgpuDeviceDestroy: _emwgpuDeviceDestroy, emwgpuInstanceRequestAdapter: _emwgpuInstanceRequestAdapter, emwgpuQueueOnSubmittedWorkDone: _emwgpuQueueOnSubmittedWorkDone, emwgpuWaitAny: _emwgpuWaitAny, environ_get: _environ_get, environ_sizes_get: _environ_sizes_get, fd_close: _fd_close, fd_read: _fd_read, fd_seek: _fd_seek, fd_write: _fd_write, ggml_jsep_alloc, ggml_jsep_clear, ggml_jsep_free, ggml_jsep_read, ggml_jsep_sync, ggml_jsep_write, ggml_webgpu_notify_async_tensor_get, wgpuAdapterGetFeatures: _wgpuAdapterGetFeatures, wgpuAdapterGetInfo: _wgpuAdapterGetInfo, wgpuAdapterGetLimits: _wgpuAdapterGetLimits, wgpuAdapterHasFeature: _wgpuAdapterHasFeature, wgpuBufferGetSize: _wgpuBufferGetSize, wgpuCommandEncoderBeginComputePass: _wgpuCommandEncoderBeginComputePass, wgpuCommandEncoderCopyBufferToBuffer: _wgpuCommandEncoderCopyBufferToBuffer, wgpuCommandEncoderFinish: _wgpuCommandEncoderFinish, wgpuCommandEncoderResolveQuerySet: _wgpuCommandEncoderResolveQuerySet, wgpuComputePassEncoderDispatchWorkgroups: _wgpuComputePassEncoderDispatchWorkgroups, wgpuComputePassEncoderEnd: _wgpuComputePassEncoderEnd, wgpuComputePassEncoderSetBindGroup: _wgpuComputePassEncoderSetBindGroup, wgpuComputePassEncoderSetPipeline: _wgpuComputePassEncoderSetPipeline, wgpuComputePipelineGetBindGroupLayout: _wgpuComputePipelineGetBindGroupLayout, wgpuDeviceCreateBindGroup: _wgpuDeviceCreateBindGroup, wgpuDeviceCreateCommandEncoder: _wgpuDeviceCreateCommandEncoder, wgpuDeviceCreateComputePipeline: _wgpuDeviceCreateComputePipeline, wgpuDeviceCreateQuerySet: _wgpuDeviceCreateQuerySet, wgpuQuerySetDestroy: _wgpuQuerySetDestroy, wgpuQueueSubmit: _wgpuQueueSubmit, wgpuQueueWriteBuffer: _wgpuQueueWriteBuffer };
  function applySignatureConversions(wasmExports2) {
    wasmExports2 = Object.assign({}, wasmExports2);
    var makeWrapper_pp = (f) => (a0) => f(a0) >>> 0;
    var makeWrapper_ppp = (f) => (a0, a1) => f(a0, a1) >>> 0;
    var makeWrapper_p = (f) => () => f() >>> 0;
    wasmExports2["malloc"] = makeWrapper_pp(wasmExports2["malloc"]);
    wasmExports2["emscripten_builtin_memalign"] = makeWrapper_ppp(wasmExports2["emscripten_builtin_memalign"]);
    wasmExports2["memalign"] = makeWrapper_ppp(wasmExports2["memalign"]);
    wasmExports2["_emscripten_stack_alloc"] = makeWrapper_pp(wasmExports2["_emscripten_stack_alloc"]);
    wasmExports2["emscripten_stack_get_current"] = makeWrapper_p(wasmExports2["emscripten_stack_get_current"]);
    return wasmExports2;
  }
  function run() {
    if (runDependencies > 0) {
      dependenciesFulfilled = run;
      return;
    }
    preRun();
    if (runDependencies > 0) {
      dependenciesFulfilled = run;
      return;
    }
    async function doRun() {
      Module2["calledRun"] = true;
      if (ABORT)
        return;
      initRuntime();
      readyPromiseResolve?.(Module2);
      Module2["onRuntimeInitialized"]?.();
      postRun();
    }
    if (Module2["setStatus"]) {
      Module2["setStatus"]("Running...");
      setTimeout(() => {
        setTimeout(() => Module2["setStatus"](""), 1);
        doRun();
      }, 1);
    } else {
      doRun();
    }
  }
  var wasmExports;
  wasmExports = await createWasm();
  run();
  if (runtimeInitialized) {
    moduleRtn = Module2;
  } else {
    moduleRtn = new Promise((resolve, reject) => {
      readyPromiseResolve = resolve;
      readyPromiseReject = reject;
    });
  }
  return moduleRtn;
}
var webllm_wasm_jsep_default;
var init_webllm_wasm_jsep = __esm(() => {
  webllm_wasm_jsep_default = Module;
});

// src/inference/jsep/command-encoder.ts
class CommandEncoderBatcher {
  device;
  maxDispatch;
  commandEncoder = null;
  passEncoder = null;
  pendingDispatchCount = 0;
  constructor(device, options) {
    this.device = device;
    this.maxDispatch = options?.maxDispatch ?? 16;
  }
  record(dispatch) {
    if (!this.commandEncoder) {
      this.commandEncoder = this.device.createCommandEncoder();
    }
    if (!this.passEncoder) {
      this.passEncoder = this.commandEncoder.beginComputePass();
    }
    this.passEncoder.setPipeline(dispatch.pipeline);
    this.passEncoder.setBindGroup(0, dispatch.bindGroup);
    this.passEncoder.dispatchWorkgroups(dispatch.dispatchX, dispatch.dispatchY, dispatch.dispatchZ);
    this.pendingDispatchCount++;
    if (this.pendingDispatchCount >= this.maxDispatch) {
      this.flush();
    }
  }
  flush() {
    if (!this.commandEncoder)
      return;
    if (this.passEncoder) {
      this.passEncoder.end();
      this.passEncoder = null;
    }
    const commands = this.commandEncoder.finish();
    this.device.queue.submit([commands]);
    this.commandEncoder = null;
    this.pendingDispatchCount = 0;
  }
  pendingCount() {
    return this.pendingDispatchCount;
  }
}

// src/inference/jsep/gpu-data-manager.ts
var SIZE_BUCKETS = [
  1 << 10,
  4 << 10,
  16 << 10,
  64 << 10,
  256 << 10,
  1 << 20,
  4 << 20,
  16 << 20,
  64 << 20,
  128 << 20
];
function pickBucket(size) {
  for (let i = 0;i < SIZE_BUCKETS.length; i++) {
    if (size <= SIZE_BUCKETS[i]) {
      return { bucketIndex: i, capacity: SIZE_BUCKETS[i] };
    }
  }
  return { bucketIndex: -1, capacity: size };
}

class GpuDataManager {
  device;
  handles = new Map;
  freeBuckets = SIZE_BUCKETS.map(() => []);
  nextHandle = 1;
  constructor(device) {
    this.device = device;
  }
  alloc(size) {
    if (size <= 0) {
      throw new Error(`GpuDataManager.alloc: invalid size ${size}`);
    }
    const { bucketIndex, capacity } = pickBucket(size);
    let record;
    if (bucketIndex >= 0) {
      const pool = this.freeBuckets[bucketIndex];
      const reused = pool.pop();
      if (reused) {
        record = reused;
      } else {
        record = {
          buffer: this.device.createBuffer({
            size: capacity,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
          }),
          size: capacity,
          bucket: bucketIndex
        };
      }
    } else {
      record = {
        buffer: this.device.createBuffer({
          size: capacity,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        }),
        size: capacity,
        bucket: -1
      };
    }
    const handle = this.nextHandle++;
    this.handles.set(handle, record);
    return handle;
  }
  free(handle) {
    const record = this.handles.get(handle);
    if (!record)
      return;
    this.handles.delete(handle);
    if (record.bucket >= 0) {
      this.freeBuckets[record.bucket].push(record);
    } else {
      record.buffer.destroy();
    }
  }
  get(handle) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.get: invalid handle ${handle}`);
    }
    return { buffer: record.buffer, size: record.size };
  }
  write(handle, offset, hostPtr, size, wasmHeap) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.write: invalid handle ${handle}`);
    }
    const view = new Uint8Array(wasmHeap, hostPtr, size);
    this.device.queue.writeBuffer(record.buffer, offset, view, 0, size);
  }
  async readAsync(handle, offset, hostPtr, size, wasmHeap) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.readAsync: invalid handle ${handle}`);
    }
    const staging = this.device.createBuffer({
      size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(record.buffer, offset, staging, 0, size);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, size);
    const mapped = new Uint8Array(staging.getMappedRange(0, size));
    const dest = new Uint8Array(wasmHeap, hostPtr, size);
    dest.set(mapped);
    staging.unmap();
    staging.destroy();
  }
  clear(handle, value, offset, size) {
    const record = this.handles.get(handle);
    if (!record) {
      throw new Error(`GpuDataManager.clear: invalid handle ${handle}`);
    }
    const scratch = new Uint8Array(size);
    if (value !== 0)
      scratch.fill(value);
    this.device.queue.writeBuffer(record.buffer, offset, scratch, 0, size);
  }
  liveHandleCount() {
    return this.handles.size;
  }
  destroy() {
    for (const record of this.handles.values()) {
      record.buffer.destroy();
    }
    this.handles.clear();
    for (const bucket of this.freeBuckets) {
      for (const record of bucket) {
        record.buffer.destroy();
      }
      bucket.length = 0;
    }
  }
}

// src/inference/jsep/ops/matmul.ts
var GGML_TYPE_F32 = 0;
var GGML_TYPE_F16 = 1;
var GGML_TYPE_Q4_0 = 2;
var GGML_TYPE_Q4_K = 12;
var TENSOR_BLOCK_I32 = 18;
var TILE_M = 16;
var TILE_N = 16;
var QK4_0 = 32;
function readDescriptor(heap32, byteOffset) {
  const base = byteOffset >> 2;
  const op = heap32[base];
  const nSrc = heap32[base + 1];
  const readBlock = (slot) => {
    const off = base + slot;
    return {
      handle: heap32[off],
      type: heap32[off + 1],
      ne: [heap32[off + 2], heap32[off + 3], heap32[off + 4], heap32[off + 5]],
      nb: [
        heap32[off + 10],
        heap32[off + 11],
        heap32[off + 12],
        heap32[off + 13]
      ]
    };
  };
  const dst = readBlock(2);
  const srcs = [];
  for (let i = 0;i < nSrc; ++i) {
    srcs.push(readBlock(2 + TENSOR_BLOCK_I32 * (1 + i)));
  }
  return { op, nSrc, dst, srcs };
}
function typeName(t) {
  switch (t) {
    case GGML_TYPE_F32:
      return "f32";
    case GGML_TYPE_F16:
      return "f16";
    case GGML_TYPE_Q4_0:
      return "q4_0";
    case GGML_TYPE_Q4_K:
      return "q4_k";
    default:
      return `t${t}`;
  }
}
function buildMatmulShader(src0Type, src1Type, dstType) {
  if (src1Type !== GGML_TYPE_F32 || dstType !== GGML_TYPE_F32) {
    throw new Error(`buildMatmulShader: unsupported (src1=${src1Type}, dst=${dstType}); ` + `Phase 2 requires src1=F32, dst=F32`);
  }
  const HEADER = `
struct Shape {
    M: u32,
    K: u32,
    N: u32,
    batch_count: u32,
    src0_row_bytes: u32,
    src1_row_bytes: u32,
    dst_row_bytes: u32,
    src0_batch_bytes: u32,
    src1_batch_bytes: u32,
    dst_batch_bytes: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> src0: array<u32>;
@group(0) @binding(1) var<storage, read> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> shape: Shape;
`;
  switch (src0Type) {
    case GGML_TYPE_F32:
      return `${HEADER}
fn load_src0(m: u32, k: u32, batch: u32) -> f32 {
    let bytes_per_elem: u32 = 4u;
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let byte_off: u32 = batch * batch_bytes + m * row_bytes + k * bytes_per_elem;
    let word_idx: u32 = byte_off / 4u;
    return bitcast<f32>(src0[word_idx]);
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_src0(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;
    case GGML_TYPE_F16:
      return `${HEADER}
fn load_src0(m: u32, k: u32, batch: u32) -> f32 {
    let bytes_per_elem: u32 = 2u;
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let byte_off: u32 = batch * batch_bytes + m * row_bytes + k * bytes_per_elem;
    let word_idx: u32 = byte_off / 4u;
    let pair: vec2<f32> = unpack2x16float(src0[word_idx]);
    // Lane within the u32 word (0 or 1) — 2 f16 per u32.
    let lane: u32 = (byte_off / 2u) & 1u;
    if (lane == 0u) {
        return pair.x;
    } else {
        return pair.y;
    }
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_src0(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;
    case GGML_TYPE_Q4_0:
      return `${HEADER}
const QK4_0: u32 = ${QK4_0}u;
const Q4_0_BYTES_PER_BLOCK: u32 = 18u;

fn load_q4_0(m: u32, k: u32, batch: u32) -> f32 {
    let row_bytes: u32 = shape.src0_row_bytes;
    let batch_bytes: u32 = shape.src0_batch_bytes;
    let row_byte_base: u32 = batch * batch_bytes + m * row_bytes;
    let block_idx: u32 = k / QK4_0;
    let in_block: u32 = k % QK4_0;
    let block_byte_base: u32 = row_byte_base + block_idx * Q4_0_BYTES_PER_BLOCK;

    // Scale: f16 at block_byte_base..+2.
    let scale_word_idx: u32 = block_byte_base / 4u;
    let scale_byte_lane: u32 = (block_byte_base / 2u) & 1u;
    let scale_pair: vec2<f32> = unpack2x16float(src0[scale_word_idx]);
    let scale: f32 = select(scale_pair.y, scale_pair.x, scale_byte_lane == 0u);

    // Nibble byte at offset (2 + (in_block % 16)); nibble lane is
    // (0 = low, 1 = high) selected by in_block / 16.
    let nibble_byte_off: u32 = block_byte_base + 2u + (in_block % 16u);
    let nibble_word_idx: u32 = nibble_byte_off / 4u;
    let nibble_byte_lane: u32 = nibble_byte_off & 3u;
    let raw_byte: u32 = (src0[nibble_word_idx] >> (nibble_byte_lane * 8u)) & 0xffu;
    let nibble: u32 = select(raw_byte >> 4u, raw_byte & 0xfu, in_block < 16u);
    return (f32(nibble) - 8.0) * scale;
}

@compute @workgroup_size(${TILE_M}, ${TILE_N}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m: u32 = gid.x;
    let n: u32 = gid.y;
    let batch: u32 = gid.z;
    if (m >= shape.M || n >= shape.N || batch >= shape.batch_count) {
        return;
    }
    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < shape.K; k = k + 1u) {
        let a: f32 = load_q4_0(m, k, batch);
        let b: f32 = src1[(batch * shape.src1_batch_bytes + n * shape.src1_row_bytes) / 4u + k];
        acc = acc + a * b;
    }
    let dst_idx: u32 = (batch * shape.dst_batch_bytes + n * shape.dst_row_bytes) / 4u + m;
    dst[dst_idx] = acc;
}
`;
    case GGML_TYPE_Q4_K:
      throw new Error("matmul Q4_K kernel: deferred to Task 7 (browser smoke covers via real weights)");
    default:
      throw new Error(`matmul: unsupported src0 type ${src0Type}`);
  }
}
var SHAPE_UNIFORM_BYTES = 12 * 4;
function buildPipeline(device, src0Type, src1Type, dstType) {
  const wgsl = buildMatmulShader(src0Type, src1Type, dstType);
  const shaderModule = device.createShaderModule({ code: wgsl });
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout]
  });
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" }
  });
}
function dispatchMatmul(ctx, desc) {
  if (desc.nSrc !== 2) {
    console.error(`dispatchMatmul: expected 2 srcs, got ${desc.nSrc}`);
    return -1;
  }
  const src0 = desc.srcs[0];
  const src1 = desc.srcs[1];
  const dst = desc.dst;
  if (src1.type !== GGML_TYPE_F32 || dst.type !== GGML_TYPE_F32) {
    console.error(`dispatchMatmul: only src1=F32, dst=F32 in Phase 2 ` + `(got src1=${src1.type}, dst=${dst.type})`);
    return -1;
  }
  const K = src0.ne[0];
  const M = src0.ne[1];
  const N = src1.ne[1];
  if (src1.ne[0] !== K || dst.ne[0] !== M || dst.ne[1] !== N) {
    console.error(`dispatchMatmul: shape mismatch — src0=[${src0.ne.join(",")}], ` + `src1=[${src1.ne.join(",")}], dst=[${dst.ne.join(",")}]`);
    return -1;
  }
  const batchCount = Math.max(1, src1.ne[2]) * Math.max(1, src1.ne[3]);
  const ndim = batchCount > 1 ? 3 : 2;
  const cacheKey = `mat-${typeName(src0.type)}-${typeName(src1.type)}-${typeName(dst.type)}-${ndim}`;
  const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
    const p = buildPipeline(device, src0.type, src1.type, dst.type);
    ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
    return p;
  });
  const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
  if (!bindGroupLayout) {
    console.error(`dispatchMatmul: missing bindGroupLayout for ${cacheKey}`);
    return -1;
  }
  const shapeBuffer = ctx.device.createBuffer({
    size: SHAPE_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  const shapeData = new Uint32Array(SHAPE_UNIFORM_BYTES / 4);
  shapeData[0] = M;
  shapeData[1] = K;
  shapeData[2] = N;
  shapeData[3] = batchCount;
  shapeData[4] = src0.nb[1];
  shapeData[5] = src1.nb[1];
  shapeData[6] = dst.nb[1];
  shapeData[7] = src0.nb[2];
  shapeData[8] = src1.nb[2];
  shapeData[9] = dst.nb[2];
  shapeData[10] = 0;
  shapeData[11] = 0;
  ctx.device.queue.writeBuffer(shapeBuffer, 0, shapeData);
  const src0Buf = ctx.dataManager.get(src0.handle).buffer;
  const src1Buf = ctx.dataManager.get(src1.handle).buffer;
  const dstBuf = ctx.dataManager.get(dst.handle).buffer;
  const bindGroup = ctx.device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: src0Buf } },
      { binding: 1, resource: { buffer: src1Buf } },
      { binding: 2, resource: { buffer: dstBuf } },
      { binding: 3, resource: { buffer: shapeBuffer } }
    ]
  });
  const dispatchX = Math.ceil(M / TILE_M);
  const dispatchY = Math.ceil(N / TILE_N);
  const dispatchZ = batchCount;
  ctx.encoderBatcher.record({
    pipeline,
    bindGroup,
    dispatchX,
    dispatchY,
    dispatchZ
  });
  return 0;
}

// src/inference/jsep/ops/rms-norm.ts
var WG_X = 1;
var WG_Y = 256;
var SHAPE_UNIFORM_BYTES2 = 4 * 4;
var RMS_NORM_WGSL = `
struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(${WG_X}, ${WG_Y})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row: u32 = gid.x;
    let col: u32 = gid.y;
    if (row >= params.rows || col >= params.cols) { return; }

    var sum_sq: f32 = 0.0;
    for (var i: u32 = 0u; i < params.cols; i = i + 1u) {
        let v: f32 = x[row * params.cols + i];
        sum_sq = sum_sq + v * v;
    }
    let inv_rms: f32 = inverseSqrt(sum_sq / f32(params.cols) + params.eps);
    let idx: u32 = row * params.cols + col;
    out[idx] = x[idx] * weight[col] * inv_rms;
}
`;
function buildPipeline2(device) {
  const shaderModule = device.createShaderModule({ code: RMS_NORM_WGSL });
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" }
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" }
      }
    ]
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout]
  });
  return device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module: shaderModule, entryPoint: "main" }
  });
}
function dispatchRmsNorm(ctx, desc, opParamsPtr, heapBuffer) {
  if (desc.nSrc !== 2) {
    console.error(`dispatchRmsNorm: expected 2 srcs, got ${desc.nSrc}`);
    return -1;
  }
  const src0 = desc.srcs[0];
  const src1 = desc.srcs[1];
  const dst = desc.dst;
  if (src0.type !== GGML_TYPE_F32 || src1.type !== GGML_TYPE_F32 || dst.type !== GGML_TYPE_F32) {
    console.error(`dispatchRmsNorm: only F32 path supported in Phase 2 ` + `(got src0=${src0.type}, src1=${src1.type}, dst=${dst.type})`);
    return -1;
  }
  const eps = new Float32Array(heapBuffer, opParamsPtr, 1)[0];
  const cols = src0.ne[0];
  const rows = src0.ne[1] * Math.max(1, src0.ne[2]) * Math.max(1, src0.ne[3]);
  if (src1.ne[0] !== cols) {
    console.error(`dispatchRmsNorm: weight length ${src1.ne[0]} != cols ${cols}`);
    return -1;
  }
  if (dst.ne[0] !== src0.ne[0] || dst.ne[1] !== src0.ne[1] || dst.ne[2] !== src0.ne[2] || dst.ne[3] !== src0.ne[3]) {
    console.error(`dispatchRmsNorm: shape mismatch — src0=[${src0.ne.join(",")}], ` + `dst=[${dst.ne.join(",")}]`);
    return -1;
  }
  const cacheKey = "rms-norm-f32";
  const pipeline = ctx.pipelineCache.getOrCreate(cacheKey, (device) => {
    const p = buildPipeline2(device);
    ctx.bindGroupLayoutCache.set(cacheKey, p.getBindGroupLayout(0));
    return p;
  });
  const bindGroupLayout = ctx.bindGroupLayoutCache.get(cacheKey);
  if (!bindGroupLayout) {
    console.error(`dispatchRmsNorm: missing bindGroupLayout for ${cacheKey}`);
    return -1;
  }
  const shapeBuffer = ctx.device.createBuffer({
    size: SHAPE_UNIFORM_BYTES2,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  const shapeU32 = new Uint32Array(SHAPE_UNIFORM_BYTES2 / 4);
  shapeU32[0] = rows;
  shapeU32[1] = cols;
  shapeU32[3] = 0;
  new Float32Array(shapeU32.buffer)[2] = eps;
  ctx.device.queue.writeBuffer(shapeBuffer, 0, shapeU32);
  const xBuf = ctx.dataManager.get(src0.handle).buffer;
  const wBuf = ctx.dataManager.get(src1.handle).buffer;
  const outBuf = ctx.dataManager.get(dst.handle).buffer;
  const bindGroup = ctx.device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: outBuf } },
      { binding: 3, resource: { buffer: shapeBuffer } }
    ]
  });
  const dispatchX = Math.ceil(rows / WG_X);
  const dispatchY = Math.ceil(cols / WG_Y);
  const dispatchZ = 1;
  ctx.encoderBatcher.record({
    pipeline,
    bindGroup,
    dispatchX,
    dispatchY,
    dispatchZ
  });
  return 0;
}

// src/inference/jsep/pipeline-cache.ts
class PipelineCache {
  device;
  cache = new Map;
  constructor(device) {
    this.device = device;
  }
  getOrCreate(key, builder) {
    const existing = this.cache.get(key);
    if (existing)
      return existing;
    const created = builder(this.device);
    this.cache.set(key, created);
    return created;
  }
  size() {
    return this.cache.size;
  }
}

// src/inference/jsep/index.ts
var STATUS_NOT_IMPLEMENTED = 1;
var GGML_OP_RMS_NORM = 25;
var GGML_OP_MUL_MAT = 29;
function installJsepCallbacks(module, device) {
  if (module.__jsep) {
    throw new Error("installJsepCallbacks: callbacks already installed on this module. " + "Call destroyJsepCallbacks(module) first if you need to re-install.");
  }
  const dataManager = new GpuDataManager(device);
  const encoderBatcher = new CommandEncoderBatcher(device);
  const pipelineCache = new PipelineCache(device);
  const bindGroupLayoutCache = new Map;
  const counters = {
    alloc: 0,
    free: 0,
    write: 0,
    read: 0,
    clear: 0,
    runOp: 0,
    sync: 0
  };
  const runtime = {
    device,
    dataManager,
    encoderBatcher,
    pipelineCache,
    bindGroupLayoutCache,
    counters
  };
  module.__jsep = runtime;
  module.jsepAlloc = (size) => {
    counters.alloc++;
    return dataManager.alloc(size);
  };
  module.jsepFree = (handle) => {
    counters.free++;
    dataManager.free(handle);
  };
  module.jsepWrite = (handle, offset, hostPtr, size) => {
    counters.write++;
    dataManager.write(handle, offset, hostPtr, size, module.HEAPU8.buffer);
  };
  module.jsepRead = (handle, offset, hostPtr, size) => {
    counters.read++;
    return dataManager.readAsync(handle, offset, hostPtr, size, module.HEAPU8.buffer);
  };
  module.jsepClear = (handle, value, offset, size) => {
    counters.clear++;
    dataManager.clear(handle, value, offset, size);
  };
  module.jsepRunOp = (descriptorPtr, _descriptorWords, opParamsPtr, _opParamsLen) => {
    counters.runOp++;
    const buf = module.HEAPU8.buffer;
    const heap32 = new Int32Array(buf, 0, buf.byteLength >>> 2);
    const desc = readDescriptor(heap32, descriptorPtr);
    const ctx = {
      device,
      dataManager,
      encoderBatcher,
      pipelineCache,
      bindGroupLayoutCache
    };
    if (desc.op === GGML_OP_MUL_MAT) {
      return dispatchMatmul(ctx, desc);
    }
    if (desc.op === GGML_OP_RMS_NORM) {
      return dispatchRmsNorm(ctx, desc, opParamsPtr, buf);
    }
    return STATUS_NOT_IMPLEMENTED;
  };
  module.jsepSync = () => {
    counters.sync++;
    encoderBatcher.flush();
  };
  return runtime;
}

// smoke-test/p2-v2-offload-probe.src.ts
function log(msg, cls = "") {
  const el = document.getElementById("log");
  if (!el)
    return;
  const line = document.createElement("div");
  if (cls)
    line.className = cls;
  line.textContent = msg;
  el.appendChild(line);
  console.log(msg);
}
async function runProbe() {
  try {
    log("[1/6] Initializing JSEP WASM module...");
    const createModule = (await Promise.resolve().then(() => (init_webllm_wasm_jsep(), exports_webllm_wasm_jsep))).default;
    window.__stderrLines = [];
    const mod = await createModule({
      printErr: (s) => {
        window.__stderrLines.push(s);
        console.error(s);
      }
    });
    log("[2/6] Acquiring WebGPU device...");
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) {
      log("no WebGPU adapter", "fail");
      return;
    }
    const device = await adapter.requestDevice();
    log("[3/6] Installing JSEP callbacks (must precede webgpu_init)...");
    installJsepCallbacks(mod, device);
    log("[4/6] Initializing ggml-webgpu backend...");
    const initStatus = await mod._webgpu_init();
    if (initStatus !== 0) {
      log(`webgpu_init returned ${initStatus}`, "fail");
      return;
    }
    log("[5/6] Snapshotting JSEP counters before probe...");
    const counter0 = {
      ...mod.__jsep?.counters ?? {}
    };
    log(`     counters@pre = ${JSON.stringify(counter0)}`);
    log("[6/6] Running webllm_synthetic_offload_probe...");
    const status = mod._webllm_synthetic_offload_probe();
    const logPtr = mod._webllm_synthetic_probe_log();
    const probeLog = mod.UTF8ToString(logPtr);
    const counter1 = {
      ...mod.__jsep?.counters ?? {}
    };
    const deltas = {};
    for (const k of Object.keys(counter1)) {
      deltas[k] = (counter1[k] ?? 0) - (counter0[k] ?? 0);
    }
    const runOpDelta = deltas.runOp ?? 0;
    log(`PROBE_STATUS = ${status}`);
    log(`PROBE_LOG =
${probeLog}`);
    log(`COUNTER_DELTAS = ${JSON.stringify(deltas)}`);
    log(`RUN_OP_DELTA = ${runOpDelta}`);
    const pass = status === 0 && runOpDelta >= 1;
    if (pass) {
      log("VERDICT: PASS — JSEP fired ≥1 MUL_MAT via offload_op.", "pass");
    } else if (status === 2) {
      log("VERDICT: NOT-APPLICABLE — JSEP not registered in this build.", "fail");
    } else if (status !== 0) {
      log(`VERDICT: FAIL — probe returned non-zero status ${status}.`, "fail");
    } else {
      log(`VERDICT: FAIL — scheduler ran but JSEP runOp delta = ${runOpDelta} (expected ≥ 1).`, "fail");
    }
    window.__probeResult = {
      status,
      log: probeLog,
      runOpDelta,
      deltas,
      counter0,
      counter1,
      pass
    };
  } catch (err) {
    const e = err;
    log(`FAIL — ${e.message}
${e.stack ?? ""}`, "fail");
  }
}
runProbe();
