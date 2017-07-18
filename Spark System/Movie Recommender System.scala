// Databricks notebook source
// MAGIC %md # Import

// COMMAND ----------

import sqlContext.implicits._
import org.apache.spark.sql._
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}

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

case class Movie(movieID: Int, title: String, genres: Seq[String])
case class User(userID: Int, gender: String, age: Int, occupation: Int, zip: String)

// COMMAND ----------

// MAGIC %md # Parse Functions

// COMMAND ----------

def parseMovie(str: String): Movie = {
  val fields = str.split(",")
  assert(fields.size == 3)
  Movie(fields(0).toInt, fields(1), Seq(fields(2)))
}

def parseUser(str: String): User = {
  val fields = str.split(",")
  assert(fields.size == 5)
  User(fields(0).toInt, fields(1).toString, fields(2).toInt, fields(3).toInt, fields(4).toString)
}

def parseRating(str: String): Rating = {
  val fields = str.split(",")
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble)
}


// COMMAND ----------

// MAGIC %md # Get Files

// COMMAND ----------

/*
// OLD DATA
val ratingsRDD = sc.textFile("/FileStore/tables/v1hi3t4p1499721624595/ratings.dat").map(parseRating)
val usersDF = sc.textFile("/FileStore/tables/v1hi3t4p1499721624595/users.dat").map(parseUser).toDF()
val moviesDF = sc.textFile("/FileStore/tables/v1hi3t4p1499721624595/movies.dat").map(parseMovie).toDF()

val ratingsDF = ratingsRDD.toDF()
*/
val ratingsRDD = sc.textFile("/FileStore/tables/p4a8vu7p1500012182568/ratings.csv").map(parseRating)
val moviesDF = sc.textFile("/FileStore/tables/p4a8vu7p1500012182568/movies.csv").map(parseMovie).toDF()
val ratingsDF = ratingsRDD.toDF()

// COMMAND ----------

// MAGIC %md # View data in SQL

// COMMAND ----------

ratingsDF.createOrReplaceTempView("ratings")
moviesDF.createOrReplaceTempView("movies")
//usersDF.createOrReplaceTempView("users")

//usersDF.printSchema()
moviesDF.printSchema()
ratingsDF.printSchema()

// COMMAND ----------

val results = sqlContext.sql(
  """select movies.title, movierates.maxr, movierates.minr, movierates.cntu
     from (SELECT ratings.product, max(ratings.rating) as maxr,
     min(ratings.rating) as minr, count(distinct user) as cntu
     FROM ratings group by ratings.product) movierates
     join movies on movierates.product=movies.movieId
     order by movierates.cntu desc""")
results.show()

// COMMAND ----------

val mostActiveUsersSchemaRDD = sqlContext.sql(
  """SELECT ratings.user, count(*) as ct from ratings
  group by ratings.user order by ct desc limit 10""")

println(mostActiveUsersSchemaRDD.collect().mkString("\n"))

val results = sqlContext.sql(
  """SELECT ratings.user, ratings.product,
  ratings.rating, movies.title FROM ratings JOIN movies
  ON movies.movieId=ratings.product
  where ratings.user=4169 and ratings.rating > 4""")

results.show

// COMMAND ----------

// MAGIC %md # Split for Tests and Training

// COMMAND ----------

val ranks = List(8, 12)
val lambdas = List(0.1, 10.0)
val numIters = List(10, 20)
var bestModel: Option[MatrixFactorizationModel] = None
var bestValidationRmse = Double.MaxValue
var bestRank = 0
var bestLambda = -1.0
var bestNumIter = -1

def computeABE(model: MatrixFactorizationModel, data: RDD[Rating]) = {
  val testUserProductRDD = testRatingsRDD.map {
    case Rating(user, product, rating) => (user, product)
  }
  val predictionsForTestRDD = model.predict(testUserProductRDD)
  val predictionsKeyedByUserProductRDD = predictionsForTestRDD.map{
    case Rating(user, product, rating) => ((user, product), rating)
  }
  val testKeyedByUserProductRDD = testRatingsRDD.map{
    case Rating(user, product, rating) => ((user, product), rating)
  }
  val testAndPredictionsJoinedRDD = testKeyedByUserProductRDD.join(predictionsKeyedByUserProductRDD)
  val falsePositives = (
    testAndPredictionsJoinedRDD.filter{
      case ((user, product), (ratingT, ratingP)) => (ratingT <= 1 && ratingP >= 4)
    })
  val meanAbsoluteError = testAndPredictionsJoinedRDD.map {
    case ((user, product), (testRating, predRating)) =>
      val err = (testRating - predRating)
      Math.abs(err)
  }.mean()
  meanAbsoluteError
}

// COMMAND ----------

val splits = ratingsRDD.randomSplit(Array(0.7, 0.2, 0.1), 0L)

val trainingRatingsRDD = splits(0).cache()
val testRatingsRDD = splits(1).cache()
val validationRatingsRDD = splits(2).cache()

val numTraining = trainingRatingsRDD.count()
val numTest = testRatingsRDD.count()

// COMMAND ----------

val model = (new ALS().setRank(20).setIterations(10).setLambda(0.01).run(trainingRatingsRDD))

// COMMAND ----------

val validationABE = computeABE(model, validationRatingsRDD)

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

val topRecsForUser = model.recommendProducts(1, 5)
val movieTitles = moviesDF.rdd.map( x => (x.getInt(0), x.getString(1)) )
val recs = topRecsForUser.map(rating => (rating.user, rating.product, rating.rating)).foreach(println)

// COMMAND ----------

topRecsForUser.take(1)

// COMMAND ----------

val topRecsForUser = model.recommendProducts(1, 5)
//val movieTitles = moviesDF.rdd.map(array => (array(0), array(1))).collect()
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

val myRatedMovieIds = 

// COMMAND ----------

import scala.util.Random

val mostRatedMovies = ratingsRDD.map(_.product)
                                .countByValue
                                .toSeq
                                .sortBy(- _._2)
                                .take(50)
                                .map(_._1)

val random = new Random(0)

// COMMAND ----------



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


