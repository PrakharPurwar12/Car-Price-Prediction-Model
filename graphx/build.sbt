name := "CarIntelGraphX"

version := "1.0.0"

scalaVersion := "2.13.16"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "4.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "4.2.0" % "provided",
  "org.apache.spark" %% "spark-graphx" % "4.2.0" % "provided"
)

javacOptions ++= Seq("-source", "17", "-target", "17")
scalacOptions ++= Seq("-deprecation", "-unchecked", "-feature")

