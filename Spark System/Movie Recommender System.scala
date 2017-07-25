// Databricks notebook source
// MAGIC %md # Import

// COMMAND ----------

import sqlContext.implicits._
import org.apache.spark.sql._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

import org.json4s._
import org.json4s.jackson.JsonMethods._

// COMMAND ----------

// MAGIC %md # Stark kafka

// COMMAND ----------

val streamingInputDF = sc.readStream
                         .format("kafka")
                         .option("kafka.bootstrap.servers", "<ip>:<port>, <ip>:<port>, <ip>:<port>")
                         .option("subscribe", "<topic>")
                         .load()

// COMMAND ----------

// MAGIC %md # Case Classes

// COMMAND ----------

// MAGIC %md # Get Files

// COMMAND ----------

val imdb_top_250 = sc.textFile("/FileStore/tables/dv8mi6s81500848697699/top250.txt").map {
  line => 
  val field = line.split("   ")
  val getName = field(3).split("  ")
  (getName(1).replaceAll("\\(.*\\)", ""))
}


// COMMAND ----------

var ratings = sc.textFile("/FileStore/tables/67pu5v311500838826920/ratings.csv").map {
  line => 
  val fields = line.split(",")
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}
val movies = sc.textFile("/FileStore/tables/ef24v7hz1500842282508/movies.csv").map {
  line =>
  val fields = line.split(",")
  (fields(0).toInt, fields(1))
}.collect.toMap

// COMMAND ----------

import org.apache.spark.sql.functions.countDistinct
ratings.toDF().agg(countDistinct("user"))

// COMMAND ----------

val ratings_json = """{
  "userID": 23,
  "movieID": 45,
  "rating": 3
}"""

val newRating = scala.util.parsing.json.JSON.parseFull(ratings_json)

newRating match {    
 case Some(m: Map[String, Double]) => {
    val a = m("userID")
    val b = m("movieID")
    val c = m("rating")
    val d = Rating(a.toInt.toInt, b.toInt, c)    
    val out = sc.parallelize(Seq(d))
    ratings = ratings.union(out)
  }
  case _ => println("Failed")
}

ratings.toDF().createOrReplaceTempView("ratings")

val result = sqlContext.sql(
  """SELECT ratings.user, ratings.product, ratings.rating
  FROM ratings
  WHERE ratings.product == 45 AND ratings.user == 23""")
result.show()

ratings

// COMMAND ----------

// MAGIC %md # Split for Tests and Training

// COMMAND ----------

val splits = ratings.randomSplit(Array(0.7, 0.2, 0.1), 0L)

val trainingRatingsRDD = splits(0).cache()
val testRatingsRDD = splits(1).cache()
val validationRatingsRDD = splits(2).cache()

//val numTraining = trainingRatingsRDD.count()
//val numTest = testRatingsRDD.count()

// COMMAND ----------

val ranks = List(8, 12)
val lambdas = List(0.1, 10.0)
val numIters = List(10, 20)
var bestModel: Option[MatrixFactorizationModel] = None
var bestValidationRmse = Double.MaxValue
var bestRank = 0
var bestLambda = -1.0
var bestNumIter = -1
val numValidation = validationRatingsRDD.count()

def computeRmse(model: MatrixFactorizationModel, data: RDD[Rating], n: Long): Double = {
  val predictions: RDD[Rating] = model.predict(data.map(x => (x.user, x.product)))
  val predictionsAndRatings = predictions.map(x => ((x.user, x.product), x.rating))
    .join(data.map(x => ((x.user, x.product), x.rating)))
    .values
  math.sqrt(predictionsAndRatings.map(x => (x._1 - x._2)*(x._1 - x._2)).reduce(_ + _) / n)
}

// COMMAND ----------

val model = (new ALS().setRank(8).setIterations(10).setLambda(10.0).run(trainingRatingsRDD))
val validationRmse = computeRmse(model, validationRatingsRDD, numValidation)
// Rank = 20, iterations = 10, lambda = 0.01, error = 84

// COMMAND ----------

def recommend(userId:Int, numOfRecommendations:Int = 10) = {
  val recommendations = model.recommendProducts(userId, numOfRecommendations)
  val filterRecommendations = recommendations
      .map(rating => (movies(rating.product)
                        .replaceAll("\\(.*\\)", "")
                        .replaceAll("\"","")))
                        .toSeq
  val jsonOut = compact(render(filterRecommendations))
}

/*
val recommendations = model.recommendProducts(23, 20)
val topRecsForUser = model.recommendProducts(111, 25)
val recs = topRecsForUser.map(rating => 
  (movies(rating.product).replaceAll("\\(.*\\)", "").replaceAll("\"",""))
).toSeq//.foreach(println)
val recsRDD = sc.parallelize(recs)
*/

// COMMAND ----------

val validationRmse = computeRmse(model, validationRatingsRDD, numValidation)
//val numUsers = ratings.map(_.user).distinct().count()

// COMMAND ----------

val validationMBE = computeABE(model, validationRatingsRDD, testRatingsRDD)

// COMMAND ----------



// COMMAND ----------

val myRatedMovieIds = 

// COMMAND ----------

for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
  val model = ALS.train(trainingRatingsRDD, rank, numIter, lambda)
  val validationABE = computeABE(model, testRatingsRDD)
  println("RMSE (validation) = " + validationABE + " for the model trained with rank = "
    + rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
  if (validationRmse < bestValidationRmse) {
    bestModel = Some(model)
    bestValidationRmse = validationRmse
    bestRank = rank
    bestLambda = lambda
    bestNumIter = numIter
  }
}

// COMMAND ----------

// MAGIC %md # Find top recommendations for a specific user

// COMMAND ----------

val topRecsForUser = model.recommendProducts(111, 25)
val recs = topRecsForUser.map(rating => 
  (movies(rating.product).replaceAll("\\(.*\\)", "").replaceAll("\"",""))
).toSeq//.foreach(println)
val recsRDD = sc.parallelize(recs)

// COMMAND ----------



compact(render(recs))

// COMMAND ----------

//topRecsForUser.take(1)
moviesDF.take(1).map( x => (x.getInt(0), x.getString(1)) )

// COMMAND ----------

val topRecsForUser = model.recommendProducts(1, 5)
val movieTitles = moviesDF.rdd.map(array => (array(0), array(1))).collect()
//val recs = topRecsForUser.map(rating => (movieTitles(rating.product), rating.rating)).foreach(println)

// COMMAND ----------

// MAGIC %md # Run tests on the accuracy

// COMMAND ----------


/*
val testUserProductRDD = testRatingsRDD.map {
  case Rating(user, product, rating) => (user, product)
}

val predictionsForTestRDD = model.predict(testUserProductRDD)

predictionsForTestRDD.take(10).mkString("\n")

val predictionsKeyedByUserProductRDD = predictionsForTestRDD.map{
  case Rating(user, product, rating) => ((user, product), rating)
}

val testKeyedByUserProductRDD = testRatingsRDD.map{
  case Rating(user, product, rating) => ((user, product), rating)
}

val testAndPredictionsJoinedRDD = testKeyedByUserProductRDD.join(predictionsKeyedByUserProductRDD)

testAndPredictionsJoinedRDD.take(3).mkString("\n")

val falsePositives = (
  testAndPredictionsJoinedRDD.filter{
    case ((user, product), (ratingT, ratingP)) => (ratingT <= 1 && ratingP >= 4)
  })

falsePositives.take(2)

falsePositives.count()

val meanAbsoluteError = testAndPredictionsJoinedRDD.map {
  case ((user, product), (testRating, predRating)) =>
    val err = (testRating - predRating)
    Math.abs(err)
}.mean()
print(meanAbsoluteError)
*/

// COMMAND ----------

// MAGIC %md # Get movie data for later

// COMMAND ----------

moviesDF.take(3).map(println)

// COMMAND ----------

class AvgCollector(val tot: Double, val cnt: Int = 1) extends Serializable {
  def ++(v: Double) = new AvgCollector(tot + v, cnt + 1)
  def combine(that: AvgCollector) = new AvgCollector(tot + that.tot, cnt + that.cnt)
  def avg = if (cnt > 0) tot / cnt else 0.0
  def count = cnt
}

val movieIDAndAverages = ratingsRDD
                      .map{ case Rating(user, product, rating) => (product, rating) }
                      .aggregateByKey(new AvgCollector(0.0,0))(_ ++ _, _ combine _ )
                      .map{ case (product, rating) => (product, rating.avg, rating.count) }

movieIDAndAverages.toDF().createOrReplaceTempView("movieIDAndAverages")

// COMMAND ----------

def elicitateRatings(movies: Seq[(Int, String)]) = {
    val prompt = "Please rate following movie (1-5(best), or 0 if not seen):"
    println(prompt)
    val ratings = movies.flatMap { x =>

      var rating: Option[Rating] = None
      var valid = false

      while (!valid) {
        print(x._2 + ": ")
        try {
          val r = Console.readInt
          if (r < 0 || r > 5) {
            println(prompt)
          } else {
            valid = true
            if (r > 0) {
              rating = Some(Rating(0, x._1, r))
            }
          }
        } catch {
          case e: Exception => println(prompt)
        }
      }

      rating match {
        case Some(r) => Iterator(r)
        case None => Iterator.empty
      }

    } //end flatMap

    if (ratings.isEmpty) {
      sys.error("No rating provided")
    } else {
      ratings
    }

  }

// COMMAND ----------

import scala.util.Random

val mostRatedMovies = ratingsRDD.map(_.product)
                                .countByValue
                                .toSeq
                                .sortBy(- _._2)
                                .take(50)
                                .map(_._1)

val random = new Random(0)
val selectedMovies = mostRatedMovies.filter(x => random.nextDouble() < 0.2)
                                    .map(x => (x, movies(x)))
                                    .toSeq

// COMMAND ----------

val selectedMovies = mostRatedMovies.filter(x => random.nextDouble() < 0.2).toDF()
                                    //.map(x => (x, moviesDF()))
                                    //.toSeq

selectedMovies.createOrReplaceTempView("selected")
selectedMovies.printSchema()
moviesDF.createOrReplaceTempView("moviesDF")
moviesDF.printSchema()
val result = sqlContext.sql(
  """
  SELECT * FROM moviesDF 
  WHERE moviesDF.movieID = selected.value
  """)
  //WHERE moviesDF.movieID = selected.value""")
  //FROM moviesDF
  //WHERE selected.value = moviesDF.movieID""")
result.show()


// COMMAND ----------


def parseRatingNewUser((field1:Int, field2:Int, field3:Int)): Rating = {
  new Rating(field1.toInt, field2.toInt, field3.toDouble)
}

//def addANewUser(newUserID:Int, arrayOfUserRating:Array[(Int, Int, Int)]) = {
//  val newUserRatingRDD = arrayOfUserRating
//  val allDataWithNewRatings = ratingsRDD.union(newUserRatingRDD)
//}

val newUserID = 0

val newUserRatings = Array(
     (0,260,4),
     (0,1,3), 
     (0,16,3), 
     (0,25,4), 
     (0,32,4), 
     (0,335,1), 
     (0,379,1), 
     (0,296,3),
     (0,858,5), 
     (0,50,4) 
    )

newUserRatings.take(1)
//val dude = newUserRatings.map(parseRatingNewUser)

//val ratingsConverted = newUserRatings.map(parseRatingNewUser)
//val newUserRatingsRDD = sc.parallelize(newUserRatings)

//val ratingsRDDTest = addANewUser(newUserID, newUserRatings)



// COMMAND ----------



// COMMAND ----------



// COMMAND ----------



// COMMAND ----------



// COMMAND ----------


