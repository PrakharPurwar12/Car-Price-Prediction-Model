package com.carintel.graph

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

case class VertexProperty(
  name: String,
  nodeType: String, // "VEHICLE", "BRAND", "FUEL", "TRANSMISSION"
  price: Double = 0.0,
  brand: String = "",
  year: Int = 0
)

case class EdgeProperty(
  relationship: String, // "MANUFACTURED_BY", "USES_FUEL", "HAS_TRANSMISSION", "SIMILAR_PRICE"
  weight: Double = 1.0
)

object VehicleGraphBuilder {

  val BRAND_OFFSET: Long = 1000000L
  val FUEL_OFFSET: Long = 2000000L
  val TRANSMISSION_OFFSET: Long = 3000000L

  def buildGraph(spark: SparkSession, vehiclesDf: DataFrame): Graph[VertexProperty, EdgeProperty] = {
    import spark.implicits._

    // 1. Vehicle Listing Vertices (ID = listing_id, positive Long)
    val vehicleVertices: RDD[(VertexId, VertexProperty)] = vehiclesDf.select(
      $"listing_id", $"full_name", $"selling_price", $"brand", $"year"
    ).rdd.map { row =>
      val id = row.getAs[Long]("listing_id")
      val name = row.getAs[String]("full_name")
      val rawPrice = row.get(row.fieldIndex("selling_price"))
      val price = rawPrice match {
        case d: java.lang.Double => d.toDouble
        case l: java.lang.Long => l.toDouble
        case i: java.lang.Integer => i.toDouble
        case other => other.toString.toDouble
      }
      val brand = row.getAs[String]("brand")
      val year = row.getAs[Int]("year")
      (id, VertexProperty(name, "VEHICLE", price, brand, year))
    }

    // 2. Brand Vertices (Deterministic Hash Offset)
    val brandList = vehiclesDf.select("brand").distinct().as[String].collect()
    val brandMap: Map[String, Long] = brandList.zipWithIndex.map { case (b, idx) =>
      b -> (BRAND_OFFSET + idx)
    }.toMap

    val brandVertices: RDD[(VertexId, VertexProperty)] = spark.sparkContext.parallelize(
      brandMap.map { case (b, id) =>
        (id, VertexProperty(b, "BRAND", brand = b))
      }.toSeq
    )

    // 3. Fuel Vertices
    val fuelList = vehiclesDf.select("fuel").distinct().as[String].collect()
    val fuelMap: Map[String, Long] = fuelList.zipWithIndex.map { case (f, idx) =>
      f -> (FUEL_OFFSET + idx)
    }.toMap

    val fuelVertices: RDD[(VertexId, VertexProperty)] = spark.sparkContext.parallelize(
      fuelMap.map { case (f, id) =>
        (id, VertexProperty(f, "FUEL"))
      }.toSeq
    )

    // 4. Transmission Vertices
    val transList = vehiclesDf.select("transmission").distinct().as[String].collect()
    val transMap: Map[String, Long] = transList.zipWithIndex.map { case (t, idx) =>
      t -> (TRANSMISSION_OFFSET + idx)
    }.toMap

    val transVertices: RDD[(VertexId, VertexProperty)] = spark.sparkContext.parallelize(
      transMap.map { case (t, id) =>
        (id, VertexProperty(t, "TRANSMISSION"))
      }.toSeq
    )

    val allVertices: RDD[(VertexId, VertexProperty)] = vehicleVertices
      .union(brandVertices)
      .union(fuelVertices)
      .union(transVertices)

    // 5. Construct Edges: Vehicle -> Brand, Vehicle -> Fuel, Vehicle -> Transmission
    val bcBrandMap = spark.sparkContext.broadcast(brandMap)
    val bcFuelMap = spark.sparkContext.broadcast(fuelMap)
    val bcTransMap = spark.sparkContext.broadcast(transMap)

    val edges: RDD[Edge[EdgeProperty]] = vehiclesDf.select(
      $"listing_id", $"brand", $"fuel", $"transmission"
    ).rdd.flatMap { row =>
      val vId = row.getAs[Long]("listing_id")
      val bName = row.getAs[String]("brand")
      val fName = row.getAs[String]("fuel")
      val tName = row.getAs[String]("transmission")

      var edgeSeq = Seq.empty[Edge[EdgeProperty]]

      bcBrandMap.value.get(bName).foreach { bId =>
        edgeSeq :+= Edge(vId, bId, EdgeProperty("MANUFACTURED_BY", 1.0))
      }
      bcFuelMap.value.get(fName).foreach { fId =>
        edgeSeq :+= Edge(vId, fId, EdgeProperty("USES_FUEL", 1.0))
      }
      bcTransMap.value.get(tName).foreach { tId =>
        edgeSeq :+= Edge(vId, tId, EdgeProperty("HAS_TRANSMISSION", 1.0))
      }

      edgeSeq
    }

    val defaultVertex = VertexProperty("UNKNOWN", "UNKNOWN")
    Graph(allVertices, edges, defaultVertex)
  }
}
