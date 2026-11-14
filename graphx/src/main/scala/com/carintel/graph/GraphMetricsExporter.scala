package com.carintel.graph

import java.io.{File, PrintWriter}
import org.apache.spark.sql.{SaveMode, SparkSession}

object GraphMetricsExporter {

  def main(args: Array[String]): Unit = {
    val inputParquet = if (args.length > 0) args(0) else "data/processed/vehicles.parquet"
    val outputParquet = if (args.length > 1) args(1) else "data/graph_metrics/graph_nodes.parquet"
    val outputJson = if (args.length > 2) args(2) else "data/graph_metrics/graph_summary.json"

    println(s"[CarIntel GraphX] Initializing SparkSession for Graph Analytics...")

    val spark = SparkSession.builder()
      .appName("CarIntel-GraphX-Engine")
      .config("spark.driver.host", "127.0.0.1")
      .config("spark.driver.bindAddress", "127.0.0.1")
      .config("spark.hadoop.fs.file.impl", "com.carintel.hadoop.WinLocalFileSystem")
      .getOrCreate()

    import spark.implicits._

    try {
      println(s"[CarIntel GraphX] Loading vehicle dataset from: $inputParquet")
      val vehiclesDf = spark.read.parquet(inputParquet)

      println(s"[CarIntel GraphX] Building Multi-Relational Graph...")
      val graph = VehicleGraphBuilder.buildGraph(spark, vehiclesDf)

      println(s"[CarIntel GraphX] Running PageRank, Connected Components & Degree Centralities...")
      val result = GraphAnalyticsRunner.runAnalytics(graph)

      println(s"[CarIntel GraphX] Graph Analytics Finished:")
      println(s"  - Total Vertices: ${result.totalVertices}")
      println(s"  - Total Edges: ${result.totalEdges}")
      println(s"  - Connected Components: ${result.numComponents}")

      println(s"[CarIntel GraphX] Exporting graph node metrics to Parquet: $outputParquet")
      val metricsDf = result.metricsRdd.toDF()
      metricsDf.write.mode(SaveMode.Overwrite).parquet(outputParquet)

      println(s"[CarIntel GraphX] Exporting summary JSON: $outputJson")
      new File(outputJson).getParentFile.mkdirs()
      
      val topNodesJson = result.topPageRankNodes.map { m =>
        s"""{"vertexId": ${m.vertexId}, "name": "${m.name.replace("\"", "\\\"")}", "nodeType": "${m.nodeType}", "pageRank": ${m.pageRank}, "totalDegree": ${m.totalDegree}, "brand": "${m.brand}"}"""
      }.mkString("[", ",", "]")

      val jsonContent =
        s"""{
           |  "total_vertices": ${result.totalVertices},
           |  "total_edges": ${result.totalEdges},
           |  "connected_components": ${result.numComponents},
           |  "top_pagerank_nodes": $topNodesJson
           |}""".stripMargin

      val pw = new PrintWriter(new File(outputJson))
      try {
        pw.write(jsonContent)
      } finally {
        pw.close()
      }

      println(s"[CarIntel GraphX] Graph execution completed successfully!")

    } finally {
      spark.stop()
    }
  }
}

