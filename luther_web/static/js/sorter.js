/**
 * High-Performance Radix Depth Sorter Controller
 * Coordinates off-thread Web Worker sorting and typed array index ordering.
 */

class RadixDepthSorter {
  constructor() {
    this.worker = null;
    this.isSorting = false;
    this.lastSortedIndices = null;
    this.onSortComplete = null;
    this.initWorker();
  }

  initWorker() {
    try {
      this.worker = new Worker("js/sort_worker.js");
      this.worker.onmessage = (e) => {
        const { type, sortedIndices } = e.data;
        if (type === "sort_done") {
          this.lastSortedIndices = new Uint32Array(sortedIndices);
          this.isSorting = false;
          if (this.onSortComplete) {
            this.onSortComplete(this.lastSortedIndices);
          }
        }
      };
    } catch (err) {
      console.warn("Web Worker initialization failed, falling back to typed array sorter:", err);
      this.worker = null;
    }
  }

  initPositions(positions, count) {
    if (this.worker) {
      this.worker.postMessage({
        type: "init",
        positions: positions.buffer,
        count: count,
      }, [positions.buffer]);
    }
  }

  requestSort(viewProjMatrixElements, count) {
    if (this.isSorting || !this.worker) return;
    this.isSorting = true;
    this.worker.postMessage({
      type: "sort",
      viewProj: viewProjMatrixElements,
      count: count,
    });
  }

  /**
   * High-Performance In-Thread 11-Bit Radix Sorter (Fallback / Testing)
   */
  static radixSortInThread(depths, count) {
    const n = count || depths.length;
    const indices = new Uint32Array(n);
    for (let i = 0; i < n; i++) indices[i] = i;

    const floatView = new Float32Array(1);
    const uintView = new Uint32Array(floatView.buffer);

    const keys = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      floatView[0] = depths[i];
      const u = uintView[0];
      keys[i] = (u & 0x80000000) ? ~u : (u ^ 0x80000000);
    }

    let src = indices;
    let dst = new Uint32Array(n);

    // 3 Passes: 11 bits, 11 bits, 10 bits
    RadixDepthSorter.pass(keys, src, dst, 0, 11, n);
    RadixDepthSorter.pass(keys, dst, src, 11, 11, n);
    RadixDepthSorter.pass(keys, src, dst, 22, 10, n);

    // Reverse for descending (back-to-front)
    const sorted = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      sorted[i] = dst[n - 1 - i];
    }
    return sorted;
  }

  static pass(keys, src, dst, shift, bits, count) {
    const mask = (1 << bits) - 1;
    const numBuckets = 1 << bits;
    const counts = new Uint32Array(numBuckets);

    for (let i = 0; i < count; i++) {
      const b = (keys[src[i]] >>> shift) & mask;
      counts[b]++;
    }

    const offsets = new Uint32Array(numBuckets);
    let total = 0;
    for (let i = 0; i < numBuckets; i++) {
      offsets[i] = total;
      total += counts[i];
    }

    for (let i = 0; i < count; i++) {
      const idx = src[i];
      const b = (keys[idx] >>> shift) & mask;
      dst[offsets[b]++] = idx;
    }
  }
}
