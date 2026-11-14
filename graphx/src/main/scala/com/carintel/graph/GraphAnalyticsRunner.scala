package com.carintel.graph

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

case class NodeMetric(
  vertexId: Long,
  name: String,
  nodeType: String,
  pageRank: Double,
  componentId: Long,
  inDegree: Int,
  outDegree: Int,
  totalDegree: Int,
  price: Double,
  brand: String,
  year: Int
)

case class GraphAnalyticsResult(
  metricsRdd: RDD[NodeMetric],
  totalVertices: Long,
  totalEdges: Long,
  numComponents: Long,
  topPageRankNodes: Array[NodeMetric]
)

object GraphAnalyticsRunner {

  def runAnalytics(
    graph: Graph[VertexProperty, EdgeProperty],
    pageRankTolerance: Double = 0.001,
    resetProb: Double = 0.15
  ): GraphAnalyticsResult = {

    val sc = graph.vertices.sparkContext

    // 1. Compute PageRank
    val prGraph: Graph[Double, Double] = graph.pageRank(pageRankTolerance, resetProb)

    // 2. Compute Connected Components
    val ccGraph: Graph[VertexId, EdgeProperty] = graph.connectedComponents()

    // 3. Compute Degree Centralities
    val inDegrees: VertexRDD[Int] = graph.inDegrees
    val outDegrees: VertexRDD[Int] = graph.outDegrees
    val degrees: VertexRDD[Int] = graph.degrees

    // 4. Join all metrics with vertex attributes
    val enrichedVertices: RDD[(VertexId, NodeMetric)] = graph.vertices
      .leftOuterJoin(prGraph.vertices)
      .leftOuterJoin(ccGraph.vertices)
      .leftOuterJoin(inDegrees)
      .leftOuterJoin(outDegrees)
      .leftOuterJoin(degrees)
      .map { case (vId, (((((vProp, prOpt), ccOpt), inDOpt), outDOpt), totDOpt)) =>
        val prop = vProp
        val pr = prOpt.getOrElse(0.0)
        val cc = ccOpt.getOrElse(vId)
        val inD = inDOpt.getOrElse(0)
        val outD = outDOpt.getOrElse(0)
        val totD = totDOpt.getOrElse(0)

        (vId, NodeMetric(
          vertexId = vId,
          name = prop.name,
          nodeType = prop.nodeType,
          pageRank = pr,
          componentId = cc,
          inDegree = inD,
          outDegree = outD,
          totalDegree = totD,
          price = prop.price,
          brand = prop.brand,
          year = prop.year
        ))
      }

    val metricsRdd = enrichedVertices.map(_._2)
    val totalV = graph.vertices.count()
    val totalE = graph.edges.count()
    val numComp = ccGraph.vertices.map(_._2).distinct().count()
    val topPR = metricsRdd.sortBy(_.pageRank, ascending = false).take(20)

    GraphAnalyticsResult(
      metricsRdd = metricsRdd,
      totalVertices = totalV,
      totalEdges = totalE,
      numComponents = numComp,
      topPageRankNodes = topPR
    )
  }
}

