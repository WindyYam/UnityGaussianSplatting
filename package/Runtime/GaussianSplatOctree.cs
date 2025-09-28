// SPDX-License-Identifier: MIT

using System.Collections.Generic;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Mathematics;
using UnityEngine;

namespace GaussianSplatting.Runtime
{
    /// <summary>
    /// Octree-based spatial acceleration structure for Gaussian splat frustum culling.
    /// Divides scene bounds into hierarchical octants for efficient culling of static splats.
    /// </summary>
    public class GaussianSplatOctree
    {
        public class OctreeNode
        {
            public Bounds bounds;
            // For leaf nodes we store original splat indices that lie within this node's bounds.
            // For internal nodes this may be null or empty.
            public List<int> splatIndices;
            // Child node indices (indices into m_Nodes). Null or empty for leaf nodes.
            public List<int> childIndices;
            public bool isLeaf;
            // Flattened leaf data (filled after build/flatten). For non-leaf nodes baseOffset/count are -1/0.
            public int baseOffset = -1; // offset into flattened leaf splat indices buffer
            public int count = 0;       // number of splats in this leaf
        }

        public struct SplatInfo
        {
            public float3 position;
            public int originalIndex;
        }

        readonly List<OctreeNode> m_Nodes = new();
        readonly List<SplatInfo> m_SplatInfos = new();
        // Legacy per-frame visible splat list (kept for fallback/debug)
        readonly List<int> m_VisibleSplatIndices = new();
        
        // Leaf indirection optimization data ----------------------------------
        readonly List<int> m_FlatLeafSplatIndices = new(); // concatenated splat indices of all leaves (including outliers leaf)
        GraphicsBuffer m_LeafSplatIndicesBuffer;            // Structured uint buffer of flattened indices
        GraphicsBuffer m_LeafMetaBuffer;                    // Structured uint2 buffer: (baseOffset, count) per node (only leaf nodes meaningful)
        readonly List<int> m_VisibleLeafNodeIndices = new(); // per-frame list of visible leaf node indices
        readonly List<int> m_VisibleLeafPrefix = new();       // prefix sums (start instance for each visible leaf)
        GraphicsBuffer m_VisibleLeafIndicesBuffer;          // per-frame visible leaf node indices
        GraphicsBuffer m_VisibleLeafPrefixBuffer;           // per-frame prefix starts
        int m_TotalVisibleFromLeaves;                       // total visible splats via leaf indirection this frame
        bool m_UseLeafIndirection = false;                   // always true for new optimization (could be heuristic later)
        int m_OthersLeafNodeIndex = -1;                     // synthetic leaf node index for outliers, -1 if none
        // ---------------------------------------------------------------------
        
        // Configuration
        int m_MaxDepth;
        int m_MaxSplatsPerLeaf;
        Bounds m_RootBounds;
        bool m_Built;

        // GPU buffer for legacy visible splat indices (updated per frame) - retained for backward compatibility
        GraphicsBuffer m_VisibleIndicesBuffer;

        // Outlier splat indices that lie outside the main root bounds (always included in culling)
        readonly List<int> m_OthersIndices = new();

        public int nodeCount => m_Nodes.Count;
        public int totalSplats => m_SplatInfos.Count;
        public bool isBuilt => m_Built;
        public GraphicsBuffer visibleIndicesBuffer => m_VisibleIndicesBuffer; // legacy path buffer
        public int visibleSplatCount { get; private set; }

        // New optimization public accessors
        public bool usingLeafIndirection => m_UseLeafIndirection && m_LeafSplatIndicesBuffer != null && m_LeafMetaBuffer != null;
        public GraphicsBuffer leafMetaBuffer => m_LeafMetaBuffer;                 // uint2(base,count) per node
        public GraphicsBuffer leafSplatIndicesBuffer => m_LeafSplatIndicesBuffer; // flattened splat indices
        public GraphicsBuffer visibleLeafIndicesBuffer => m_VisibleLeafIndicesBuffer; // per-frame visible leaf node indices
        public GraphicsBuffer visibleLeafPrefixBuffer => m_VisibleLeafPrefixBuffer;   // per-frame prefix start array
        public int visibleLeafCount => m_VisibleLeafNodeIndices.Count;
        public int totalVisibleLeafMappedSplats => m_TotalVisibleFromLeaves;      // same as visibleSplatCount when using indirection

        /// <summary>
        /// Initialize octree parameters. Call this before building.
        /// </summary>
        /// <param name="maxDepth">Maximum tree depth (typically 4-6)</param>
        /// <param name="maxSplatsPerLeaf">Maximum splats per leaf node (typically 64-256)</param>
        public void Initialize(int maxDepth = 5, int maxSplatsPerLeaf = 128)
        {
            m_MaxDepth = maxDepth;
            m_MaxSplatsPerLeaf = maxSplatsPerLeaf;
        }

        /// <summary>
        /// Build octree from splat position data and bounds.
        /// </summary>
        public void Build(NativeArray<float3> splatPositions, Bounds sceneBounds)
        {
            Clear();
            
            if (splatPositions.Length == 0)
            {
                Debug.LogWarning("GaussianSplatOctree.Build: No splat positions provided");
                return;
            }
            
            Debug.Log($"Building octree with {splatPositions.Length} splats, bounds: {sceneBounds}");
            
            // Compute center of mass and identify 95% closest splats
            int total = splatPositions.Length;
            float3 com = float3.zero;
            for (int i = 0; i < total; i++)
                com += splatPositions[i];
            com /= total;

            var distList = new List<(int idx, float d)>();
            distList.Capacity = total;
            for (int i = 0; i < total; i++)
            {
              float distance = math.distance(splatPositions[i], com);
              distList.Add((i, distance));
            }
            distList.Sort((a, b) => a.d.CompareTo(b.d));

            // Reorder m_SplatInfos so that the closest 95% are first, others last
            int inCount = Mathf.CeilToInt(total * 0.99f);
            inCount = Mathf.Clamp(inCount, 1, total);
            int othersCount = total - inCount;
            
            m_SplatInfos.Clear();
            m_SplatInfos.Capacity = total;

            for (int i = 0; i < total; i++)
            {
              int src = distList[i].idx;
              m_SplatInfos.Add(new SplatInfo { position = splatPositions[src], originalIndex = src });
            }

            // Create root bounds based on the inCount splats (centered on center-of-mass)
            Bounds rootBounds;
            if (inCount > 0)
            {
                float3 min = m_SplatInfos[0].position;
                float3 max = m_SplatInfos[0].position;
                for (int i = 1; i < inCount; i++)
                {
                    min = math.min(min, m_SplatInfos[i].position);
                    max = math.max(max, m_SplatInfos[i].position);
                }
                rootBounds = new Bounds((max + min) * 0.5f, max - min);
            }
            else
            {
                // Fallback to provided scene bounds
                rootBounds = sceneBounds;
            }

            m_RootBounds = rootBounds;

            // Build tree recursively using only the in-root splats
            m_Nodes.Clear();
            
            // Create root node covering the in-root splats
            var rootNode = new OctreeNode
            {
                bounds = m_RootBounds,
                splatIndices = null,
                childIndices = null,
                isLeaf = false
            };
            m_Nodes.Add(rootNode);

            // Build recursively starting from root (only for the in-root partition)
            var rootSplatList = new List<int>(inCount);
            for (int i = 0; i < inCount; i++) rootSplatList.Add(i); // indices into m_SplatInfos
            BuildRecursive(0, 0, rootSplatList);

            // Handle remaining outliers: put their original indices into m_OthersIndices list
            m_OthersIndices.Clear();
            if (othersCount > 0)
            {
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = m_SplatInfos[inCount + i].originalIndex;
                    m_OthersIndices.Add(orig);
                }
            }

            // Flatten leaves (including synthetic outliers leaf if needed)
            FlattenLeavesIncludeOutliers();

            m_Built = true;

            Debug.Log($"Octree build completed: {m_Nodes.Count} total nodes, others={m_OthersIndices.Count}, flattenedLeafIndices={m_FlatLeafSplatIndices.Count}");
        }

        void BuildRecursive(int nodeIndex, int depth, List<int> splatList)
         {
            var node = m_Nodes[nodeIndex];
            
            // Check termination conditions
            if (depth >= m_MaxDepth || splatList.Count <= m_MaxSplatsPerLeaf)
            {
                // Make this a leaf node and store original indices for this leaf
                node.isLeaf = true;
                node.splatIndices = new List<int>(splatList.Count);
                for (int i = 0; i < splatList.Count; i++)
                {
                    int infoIdx = splatList[i];
                    if (infoIdx < 0 || infoIdx >= m_SplatInfos.Count)
                    {
                        Debug.LogError($"Octree leaf node splat info index out of bounds: {infoIdx} >= {m_SplatInfos.Count}");
                        continue;
                    }
                    node.splatIndices.Add(m_SplatInfos[infoIdx].originalIndex);
                }

                m_Nodes[nodeIndex] = node;
                return;
            }

             // Create 8 child nodes
             var center = node.bounds.center;
             var size = node.bounds.size * 0.5f;
             
             node.childIndices = new List<int>(8);
             node.isLeaf = false;
             m_Nodes[nodeIndex] = node;
             
             // Create child bounds
             var childBounds = new Bounds[8];
             for (int i = 0; i < 8; i++)
             {
                 var offset = new Vector3(
                     (i & 1) != 0 ? size.x * 0.5f : -size.x * 0.5f,
                     (i & 2) != 0 ? size.y * 0.5f : -size.y * 0.5f,
                     (i & 4) != 0 ? size.z * 0.5f : -size.z * 0.5f
                 );
                 childBounds[i] = new Bounds(center + offset, size);
             }

             // Distribute splats to children
            var childSplatsIdx = new List<int>[8];
            for (int i = 0; i < 8; i++) childSplatsIdx[i] = new List<int>();

            // Assign splats (using splatList which holds indices into m_SplatInfos) to child nodes
            for (int ii = 0; ii < splatList.Count; ii++)
            {
                int infoIdx = splatList[ii];
                if (infoIdx < 0 || infoIdx >= m_SplatInfos.Count)
                {
                    Debug.LogError($"Octree splat distribution info index out of bounds: {infoIdx} >= {m_SplatInfos.Count}");
                    continue;
                }

                var splat = m_SplatInfos[infoIdx];

                int childIndex = 0;
                if (splat.position.x > center.x) childIndex |= 1;
                if (splat.position.y > center.y) childIndex |= 2;
                if (splat.position.z > center.z) childIndex |= 4;

                childSplatsIdx[childIndex].Add(infoIdx);
            }

             // Create child nodes
            for (int i = 0; i < 8; i++)
            {
                var childNode = new OctreeNode
                {
                    bounds = childBounds[i],
                    splatIndices = null,
                    childIndices = null,
                    isLeaf = childSplatsIdx[i].Count == 0
                };

                int childNodeIndex = m_Nodes.Count;
                m_Nodes.Add(childNode);

                // Register child index with parent
                node.childIndices.Add(childNodeIndex);
                // Update parent reference in the global list (node is a reference type)
                m_Nodes[nodeIndex] = node;
                
                // Recursively build child only if it has splats
                if (childSplatsIdx[i].Count > 0)
                {
                    BuildRecursive(childNodeIndex, depth + 1, childSplatsIdx[i]);
                }
            }
         }

        // Flatten leaf splat indices into contiguous array + meta (base,count). Add synthetic leaf for outliers.
        void FlattenLeavesIncludeOutliers()
        {
            m_FlatLeafSplatIndices.Clear();
            // Reserve approximate capacity
            m_FlatLeafSplatIndices.Capacity = Mathf.Max(m_FlatLeafSplatIndices.Capacity, totalSplats);
            
            // Iterate nodes, assign base/count
            for (int i = 0; i < m_Nodes.Count; i++)
            {
                var node = m_Nodes[i];
                if (node.isLeaf && node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    node.baseOffset = m_FlatLeafSplatIndices.Count;
                    node.count = node.splatIndices.Count;
                    m_FlatLeafSplatIndices.AddRange(node.splatIndices);
                    m_Nodes[i] = node;
                }
                else
                {
                    node.baseOffset = -1;
                    node.count = 0;
                    m_Nodes[i] = node;
                }
            }

            // Synthetic leaf for outliers (always considered visible)
            m_OthersLeafNodeIndex = -1;
            if (m_OthersIndices.Count > 0)
            {
                var othersNode = new OctreeNode
                {
                    bounds = new Bounds(m_RootBounds.center, Vector3.zero),
                    isLeaf = true,
                    splatIndices = null, // we rely only on flattened data
                    childIndices = null,
                    baseOffset = m_FlatLeafSplatIndices.Count,
                    count = m_OthersIndices.Count
                };
                m_FlatLeafSplatIndices.AddRange(m_OthersIndices);
                m_OthersLeafNodeIndex = m_Nodes.Count;
                m_Nodes.Add(othersNode);
            }

            // Upload static GPU buffers
            UploadStaticLeafBuffers();
        }

        void UploadStaticLeafBuffers()
        {
            // Leaf meta buffer (uint2 per node: base, count) - allocate for all nodes for easy indexing
            int nodeCountLocal = m_Nodes.Count;
            var metaNative = new NativeArray<uint2>(nodeCountLocal, Allocator.Temp, NativeArrayOptions.ClearMemory);
            for (int i = 0; i < nodeCountLocal; i++)
            {
                var n = m_Nodes[i];
                if (n.baseOffset >= 0 && n.count > 0)
                {
                    metaNative[i] = new uint2((uint)n.baseOffset, (uint)n.count);
                }
            }

            m_LeafMetaBuffer?.Dispose();
            m_LeafMetaBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, nodeCountLocal, UnsafeUtility.SizeOf<uint2>())
            {
                name = "GaussianSplatLeafMeta"
            };
            m_LeafMetaBuffer.SetData(metaNative);
            metaNative.Dispose();

            // Flattened indices buffer
            int flatCount = m_FlatLeafSplatIndices.Count;
            if (flatCount > 0)
            {
                var flatNative = new NativeArray<uint>(flatCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
                for (int i = 0; i < flatCount; i++) flatNative[i] = (uint)m_FlatLeafSplatIndices[i];
                m_LeafSplatIndicesBuffer?.Dispose();
                m_LeafSplatIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, flatCount, sizeof(uint))
                {
                    name = "GaussianSplatLeafFlattenedIndices"
                };
                m_LeafSplatIndicesBuffer.SetData(flatNative);
                flatNative.Dispose();
            }
            else
            {
                m_LeafSplatIndicesBuffer?.Dispose();
                m_LeafSplatIndicesBuffer = null;
            }
        }

         /// <summary>
         /// Perform frustum culling and update visible splat (or leaf) indices.
         /// Returns number of visible splats.
         /// </summary>
         public int CullFrustum(Camera camera)
         {
             if (!m_Built)
                 return 0;

             // Decide path (currently always use leaf indirection if available)
             if (usingLeafIndirection)
             {
                 CullLeaves(camera);
                 visibleSplatCount = m_TotalVisibleFromLeaves;
                 return visibleSplatCount;
             }

             // Legacy path (should rarely execute once leaf indirection enabled)
             m_VisibleSplatIndices.Clear();
             var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
             CullNodeRecursiveCollectSplats(0, frustumPlanes);
             // Always include 'others' outlier splats
             if (m_OthersIndices.Count > 0)
                 m_VisibleSplatIndices.AddRange(m_OthersIndices);
             visibleSplatCount = m_VisibleSplatIndices.Count;
             UpdateVisibleIndicesBuffer();
             return visibleSplatCount;
         }

        void CullLeaves(Camera camera)
        {
            m_VisibleLeafNodeIndices.Clear();
            m_VisibleLeafPrefix.Clear();
            m_TotalVisibleFromLeaves = 0;
            if (m_Nodes.Count == 0) return;
            var planes = GeometryUtility.CalculateFrustumPlanes(camera);
            // Traverse all leaf nodes (simple linear scan acceptable since number of leaves << splats)
            for (int i = 0; i < m_Nodes.Count; i++)
            {
                var node = m_Nodes[i];
                if (!node.isLeaf || node.count <= 0) continue;
                if (i == m_OthersLeafNodeIndex) continue; // handle separately (always visible)
                if (!GeometryUtility.TestPlanesAABB(planes, node.bounds)) continue;
                m_VisibleLeafNodeIndices.Add(i);
            }
            // Always append synthetic outliers leaf (if any)
            if (m_OthersLeafNodeIndex >= 0)
                m_VisibleLeafNodeIndices.Add(m_OthersLeafNodeIndex);
            // Build prefix and total
            int running = 0;
            for (int i = 0; i < m_VisibleLeafNodeIndices.Count; i++)
            {
                m_VisibleLeafPrefix.Add(running);
                var node = m_Nodes[m_VisibleLeafNodeIndices[i]];
                running += node.count;
            }
            m_TotalVisibleFromLeaves = running;
            UploadVisibleLeafBuffers();
        }

        void UploadVisibleLeafBuffers()
        {
            // Upload visible leaf indices
            int leafCount = m_VisibleLeafNodeIndices.Count;
            if (leafCount == 0)
            {
                m_VisibleLeafIndicesBuffer?.Dispose(); m_VisibleLeafIndicesBuffer = null;
                m_VisibleLeafPrefixBuffer?.Dispose(); m_VisibleLeafPrefixBuffer = null;
                return;
            }
            // Ensure/resize buffers
            if (m_VisibleLeafIndicesBuffer == null || m_VisibleLeafIndicesBuffer.count < leafCount)
            {
                m_VisibleLeafIndicesBuffer?.Dispose();
                m_VisibleLeafIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.NextPowerOfTwo(leafCount), sizeof(uint))
                {
                    name = "GaussianSplatVisibleLeafIndices"
                };
            }
            if (m_VisibleLeafPrefixBuffer == null || m_VisibleLeafPrefixBuffer.count < leafCount)
            {
                m_VisibleLeafPrefixBuffer?.Dispose();
                m_VisibleLeafPrefixBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, Mathf.NextPowerOfTwo(leafCount), sizeof(uint))
                {
                    name = "GaussianSplatVisibleLeafPrefix"
                };
            }
            var visLeafNative = new NativeArray<uint>(leafCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            var prefixNative = new NativeArray<uint>(leafCount, Allocator.Temp, NativeArrayOptions.UninitializedMemory);
            for (int i = 0; i < leafCount; i++)
            {
                visLeafNative[i] = (uint)m_VisibleLeafNodeIndices[i];
                prefixNative[i] = (uint)m_VisibleLeafPrefix[i];
            }
            m_VisibleLeafIndicesBuffer.SetData(visLeafNative, 0, 0, leafCount);
            m_VisibleLeafPrefixBuffer.SetData(prefixNative, 0, 0, leafCount);
            visLeafNative.Dispose();
            prefixNative.Dispose();
        }

         void CullNodeRecursiveCollectSplats(int nodeIndex, Plane[] frustumPlanes)
        {
            if (nodeIndex >= m_Nodes.Count)
                return;

            var node = m_Nodes[nodeIndex];

            // Test node bounds against frustum
            if (!GeometryUtility.TestPlanesAABB(frustumPlanes, node.bounds))
                return; // Node is outside frustum

            if (node.isLeaf)
            {
                // Add all splats in this leaf to visible list (skip empty leaves)
                if (node.splatIndices != null)
                {
                    for (int i = 0; i < node.splatIndices.Count; i++)
                    {
                        m_VisibleSplatIndices.Add(node.splatIndices[i]);
                    }
                }
            }
            else
            {
                // Recursively test child nodes - only traverse non-empty children
                if (node.childIndices != null)
                {
                    foreach (var childIndex in node.childIndices)
                    {
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
                            if ((childNode.splatIndices != null && childNode.splatIndices.Count > 0) || !childNode.isLeaf)
                            {
                                CullNodeRecursiveCollectSplats(childIndex, frustumPlanes);
                            }
                        }
                    }
                }
            }
         }

         void UpdateVisibleIndicesBuffer()
         {
             if (visibleSplatCount == 0)
                return;
            
            // Ensure buffer is large enough
            int requiredSize = visibleSplatCount;
            if (m_VisibleIndicesBuffer == null || m_VisibleIndicesBuffer.count < requiredSize)
            {
                m_VisibleIndicesBuffer?.Dispose();
                // Allocate with some extra space to avoid frequent reallocations
                int bufferSize = Mathf.NextPowerOfTwo(requiredSize);
                m_VisibleIndicesBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, bufferSize, sizeof(uint))
                {
                    name = "GaussianSplatVisibleIndices"
                };
            }
            
            // Convert to uint array and upload visible indices
            var nativeArray = new NativeArray<uint>(visibleSplatCount, Allocator.Temp);
            for (int i = 0; i < visibleSplatCount; i++)
            {
                nativeArray[i] = (uint)m_VisibleSplatIndices[i];
            }
            m_VisibleIndicesBuffer.SetData(nativeArray, 0, 0, visibleSplatCount);
            nativeArray.Dispose();
        }

        /// <summary>
        /// Get debug information about octree structure.
        /// </summary>
        public void GetDebugInfo(out int leafNodes, out int maxDepthReached, out int maxSplatsInLeaf)
        {
            leafNodes = 0;
            maxDepthReached = 0;
            maxSplatsInLeaf = 0;
            
            GetDebugInfoRecursive(0, 0, ref leafNodes, ref maxDepthReached, ref maxSplatsInLeaf);
        }
        
        void GetDebugInfoRecursive(int nodeIndex, int depth, ref int leafNodes, ref int maxDepth, ref int maxSplats)
        {
            if (nodeIndex >= m_Nodes.Count)
                return;

            var node = m_Nodes[nodeIndex];
            maxDepth = Mathf.Max(maxDepth, depth);

            if (node.isLeaf)
            {
                // Only count non-empty leaves
                if ((node.splatIndices != null && node.splatIndices.Count > 0) || node.count > 0)
                {
                    leafNodes++;
                    int c = node.count > 0 ? node.count : (node.splatIndices?.Count ?? 0);
                    maxSplats = Mathf.Max(maxSplats, c);
                }
            }
            else
            {
                // Traverse registered child indices
                if (node.childIndices != null)
                {
                    foreach (var childIndex in node.childIndices)
                    {
                        if (childIndex < m_Nodes.Count)
                        {
                            GetDebugInfoRecursive(childIndex, depth + 1, ref leafNodes, ref maxDepth, ref maxSplats);
                        }
                    }
                }
            }
        }

        /// <summary>
         /// Draw wireframe boxes for each non-empty leaf node. Call this from a MonoBehaviour's OnDrawGizmos or OnDrawGizmosSelected.
         /// </summary>
         public void DrawLeafBoundsGizmos(Color color)
         {
             if (!m_Built || m_Nodes.Count == 0)
                 return;

             var prev = Gizmos.color;
             Gizmos.color = color;

             for (int i = 0; i < m_Nodes.Count; i++)
             {
                 var node = m_Nodes[i];
                 if (!node.isLeaf)
                     continue;

                 // Skip empty leaves
                 int c = node.count > 0 ? node.count : (node.splatIndices?.Count ?? 0);
                 if (c <= 0)
                     continue;

                 Gizmos.DrawWireCube(node.bounds.center, node.bounds.size);
             }

             Gizmos.color = prev;
         }

         public void Clear()
         {
             m_Nodes.Clear();
             m_SplatInfos.Clear();
             m_VisibleSplatIndices.Clear();
             m_FlatLeafSplatIndices.Clear();
             m_VisibleLeafNodeIndices.Clear();
             m_VisibleLeafPrefix.Clear();
             m_VisibleIndicesBuffer?.Dispose();
             m_VisibleIndicesBuffer = null;
             m_LeafSplatIndicesBuffer?.Dispose();
             m_LeafSplatIndicesBuffer = null;
             m_LeafMetaBuffer?.Dispose();
             m_LeafMetaBuffer = null;
             m_VisibleLeafIndicesBuffer?.Dispose();
             m_VisibleLeafIndicesBuffer = null;
             m_VisibleLeafPrefixBuffer?.Dispose();
             m_VisibleLeafPrefixBuffer = null;
             visibleSplatCount = 0;
             m_TotalVisibleFromLeaves = 0;
             m_Built = false;
             m_OthersIndices.Clear();
             m_OthersLeafNodeIndex = -1;
         }

         public void Dispose()
         {
             Clear();
         }
    }
}
