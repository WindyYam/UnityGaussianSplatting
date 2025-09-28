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
        }

        public struct SplatInfo
        {
            public float3 position;
            public int originalIndex;
        }

        readonly List<OctreeNode> m_Nodes = new();
        readonly List<SplatInfo> m_SplatInfos = new();
        // Note: per-node splat indices are stored inside OctreeNode.splatIndices for leaf nodes.
         readonly List<int> m_VisibleSplatIndices = new();
         
         // Configuration
         int m_MaxDepth;
         int m_MaxSplatsPerLeaf;
         Bounds m_RootBounds;
         bool m_Built;

        // GPU buffer for visible splat indices (updated per frame/N frames)
        GraphicsBuffer m_VisibleIndicesBuffer;

        // Outlier splat indices that lie outside the main root bounds (always included in culling)
        readonly List<int> m_OthersIndices = new();

        public int nodeCount => m_Nodes.Count;
        public int totalSplats => m_SplatInfos.Count;
        public bool isBuilt => m_Built;
        public GraphicsBuffer visibleIndicesBuffer => m_VisibleIndicesBuffer;
        public int visibleSplatCount { get; private set; }

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
        public void Build(NativeArray<float3> splatPositions, Bounds sceneBounds, float splatPercent)
        {
            Clear();
            // m_OthersNodeIndex removed - use m_OthersIndices list instead
            
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

            // Reorder m_SplatInfos so that the closest part are first, others last
            int inCount = Mathf.CeilToInt(total * splatPercent);
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

            // Handle remaining outliers: put their original indices into m_SplatIndices and track them in m_OthersIndices
            m_OthersIndices.Clear();
            if (othersCount > 0)
            {
                for (int i = 0; i < othersCount; i++)
                {
                    int orig = m_SplatInfos[inCount + i].originalIndex;
                     m_OthersIndices.Add(orig);
                }
            }

            m_Built = true;

            Debug.Log($"Octree build completed: {m_Nodes.Count} total nodes, others={m_OthersIndices.Count}");
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

         /// <summary>
         /// Perform frustum culling and update visible splat indices.
         /// Returns number of visible splats.
         /// </summary>
         public int CullFrustum(Camera camera)
         {
             if (!m_Built)
                 return 0;

             m_VisibleSplatIndices.Clear();
             
             // Extract frustum planes from camera
             var frustumPlanes = GeometryUtility.CalculateFrustumPlanes(camera);
             
             // Traverse octree and collect visible splats
            CullNodeRecursive(0, frustumPlanes);

             // Always include 'others' outlier splats
             if (m_OthersIndices.Count > 0)
             {
                 // Fast bulk append outlier indices
                 m_VisibleSplatIndices.AddRange(m_OthersIndices);
             }
             
             visibleSplatCount = m_VisibleSplatIndices.Count;
             
             // Debug log culling performance
            //float cullingRatio = (float)visibleSplatCount / Mathf.Max(1, m_SplatInfos.Count);
            //Debug.Log($"Octree culling: {visibleSplatCount}/{m_SplatInfos.Count} splats visible ({cullingRatio:P1})");
             
             // Update GPU buffer
             UpdateVisibleIndicesBuffer();
             
             return visibleSplatCount;
         }

         void CullNodeRecursive(int nodeIndex, Plane[] frustumPlanes)
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
                // Traverse registered child indices
                if (node.childIndices != null)
                {
                    foreach (var childIndex in node.childIndices)
                    {
                        if (childIndex < m_Nodes.Count)
                        {
                            var childNode = m_Nodes[childIndex];
                            if ((childNode.splatIndices != null && childNode.splatIndices.Count > 0) || !childNode.isLeaf)
                            {
                                CullNodeRecursive(childIndex, frustumPlanes);
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
                if (node.splatIndices != null && node.splatIndices.Count > 0)
                {
                    leafNodes++;
                    maxSplats = Mathf.Max(maxSplats, node.splatIndices.Count);
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
                 if (node.splatIndices == null || node.splatIndices.Count <= 0)
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
             m_VisibleIndicesBuffer?.Dispose();
             m_VisibleIndicesBuffer = null;
             visibleSplatCount = 0;
             m_Built = false;
             m_OthersIndices.Clear();
         }

         public void Dispose()
         {
             Clear();
         }
    }
}
