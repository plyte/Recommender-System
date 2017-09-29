// Databricks notebook source
// MAGIC %md # Widgets for input

// COMMAND ----------

import java.lang.NumberFormatException

dbutils.widgets.text("userId", "0")
val widget = dbutils.widgets.get("userId")

def toInt(s: String): Option[Int] = {
  try {
    Some(s.toInt)
  } catch {
    case e: Exception => None
  }
}

// COMMAND ----------

// MAGIC %md # Mount

// COMMAND ----------

import java.rmi.RemoteException

def mount() {
  val AccessKey = "PLACEHOLDER"
  val SecretKey = "PLACEHOLDER"
  val EncodedSecretKey = SecretKey.replace("/", "%2F")
  val AwsBucketName = "movies-rating"
  val MountName = "s3files"
  try {
    //dbutils.fs.unmount(s"/mnt/$MountName")
    dbutils.fs.mount(s"s3a://$AccessKey:$EncodedSecretKey@$AwsBucketName", s"/mnt/$MountName")
  }
  catch {
    case e: RemoteException => { println("===drive already mounted===") } 
  }
  finally {
    
  }
}

mount()



// COMMAND ----------

// MAGIC %md # Classes

// COMMAND ----------

// MAGIC %md ### Imports

// COMMAND ----------

import sqlContext.implicits._
import org.apache.spark.sql._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

import org.json4s._
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import java.util.Properties
import awscala._, dynamodbv2._

// COMMAND ----------

// MAGIC %md ### Recommender System

// COMMAND ----------

/*
 * Where all the calucations for the recommender system takes place
 *
 * @param userId: Input to the class that contains the intended user for evaulation
 * @param ratings: The database containing the userId, movieId, and rating on each line
 * @param movies: The database containing the movies for use in taking the movieId of the ratings database and mapping it to a string
 */
class RecommenderSystem(userId:Int, ratings:RDD[Rating], movies:Map[Int, String]) {
  
  /*
   * Trains a mode of the datbase passed from the Database class
   * @return: Returns a trained model of the database passed by the Database class
   */
  def train():MatrixFactorizationModel = {
    println("===Training===")
    val model = (new ALS().setRank(20).setIterations(10).setLambda(0.01).run(ratings))
    model
  }
  
  /*
   * Recommends a list of movies based on the user specified 
   *
   * @model: Input the MatrixFactorizationMode that have been output by the train() function
   * @numOfRecommendations: Specify the number of recommendations output by the model 
   * @return: Returns a JSON formatted list of movie titles
   */
  def recommendProducts(model:MatrixFactorizationModel, numOfRecommendations:Int = 25) = {
    val recommendations = model.recommendProducts(userId, numOfRecommendations)
    val filterRecommendations = recommendations
        .map(rating => (movies(rating.product)
                          .replaceAll("\\(.*\\)", "")
                          .replaceAll("\"","")))
                          .toSeq
    filterRecommendations
  }
}

/*
 * A class that contains the ratings and movies databases for use in the RecommenderSystem class
 */
class Database() {
  
  val MattAWSAccessKeyId = "PLACEHOLDER"
  val MattAWSSecretKey = "PLACEHOLDER"
  
  /**
    * Aquires the ratings database which contains the userId, movieId, rating
    * toParse:RDD[Row]
    * @return returns an RDD[Rating] that contains a Ratings class with 3 fields (userId, movieId, rating)
    */
  def loadRatings() = {
    val connectionProperties = new Properties()
    connectionProperties.put("user", "PLACEHOLDER")
    connectionProperties.put("password", "PLACEHOLDER")
    
    val jdbcDatabase = "PLACEHOLDER"
    val jdbcPort = 1234
    val jdbcHostname = "PLACEHOLDER"

    val jdbc_url = s"jdbc:mysql://${jdbcHostname}:${jdbcPort}/${jdbcDatabase}"

    val toParse = spark.read.jdbc(jdbc_url, "ratings", connectionProperties).rdd
    val ratings = toParse.map {
      row =>
      Rating(row.getInt(0), row.getInt(1), row.getDouble(2))
    }
    ratings
  }
  
  /**
    * Aquires the movies database which contains the movieId and movieName
    *
    * @param moviesPath: The path where the movies database is located
    * @return returns a Map[Int, String] that contains a tuple of (movieId, movieName)
    */
  def loadMovies(moviesPath:String = "") = {
    val movies = sc.textFile(moviesPath).map {
      line =>
      val fields = line.split(",")
      (fields(0).toInt, fields(1))
    }.collect.toMap
    movies
  }
  
  def pushRecommendations(recommendations:Seq[String], userId:Integer) = {
    val timestamp: Long = System.currentTimeMillis / 1000
    implicit val dynamoDB = DynamoDB.apply(MattAWSAccessKeyId, MattAWSSecretKey)(Region.US_EAST_1)
    val table: Table = dynamoDB.table("rm_predictions").get
    table.put(userId, "Movies" -> recommendations, "timestamp" -> timestamp)
  }
}


// COMMAND ----------

// MAGIC %md # Execute function

// COMMAND ----------

val parseWidget = toInt(widget)
val userId = parseWidget.getOrElse(0)

if (userId != 0) {
  val db = new Database()
  val ratings = db.loadRatings()
  val movies = db.loadMovies("/mnt/s3files/movies-small.csv")
  val rS = new RecommenderSystem(userId, ratings, movies)
  val trainedModel = rS.train()
  val recommendations = rS.recommendProducts(trainedModel)
  db.pushRecommendations(recommendations, userId)
}

// COMMAND ----------


