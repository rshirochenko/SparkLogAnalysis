/* Shirochenko Roman. Home Assignment 2. Big Data Course. */
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.optimization.L1Updater

import scala.util.parsing.combinator._


abstract class LogLine extends java.io.Serializable
case class AppSummary(timestamp: String, app: String, name: String, user: String, state:String,
  url:String, host: String, startTime: String, endTime: String, finalStatus: String) extends LogLine
case class CapacitySchedulery(timestamp: String, app:String, noname: String, event: String) extends LogLine
case class UnknownLine() extends LogLine


// Yarn Log Parser
object LogP extends RegexParsers with java.io.Serializable {
  def logline: Parser[LogLine] = (
    timestamp~"INFO org.apache.hadoop.yarn.server.resourcemanager.RMAppManager$ApplicationSummary: appId=application_"~ident
        ~",name="~identW  
        ~",user="~ident
        ~",queue=default,state="~ident
        ~",trackingUrl="~url
        ~",appMasterHost="~ident
        ~".icdatacluster2,startTime="~ident
        ~",finishTime="~ident
        ~",finalStatus="~ident ^^ {
       case t~_~app~_~name~_~user~_~state~_~url~_~host~_~stime~_~etime~_~finalStatus =>
         AppSummary(t, app, name, user, state, url, host, stime, etime, finalStatus)
    }
  | timestamp~"INFO org.apache.hadoop.yarn.server.resourcemanager.scheduler.capacity.CapacityScheduler: Application attempt appattempt_"~clean_apl
       ~"_000001 released container "~before_event
       ~"event: "~ident ^^ {
       case timestamp~_~app~_~noname~_~event =>
         CapacitySchedulery(timestamp, app, noname, event)
    }

  )

  val ident: Parser[String] = "[A-Za-z0-9_]+".r
  val identW: Parser[String] = "[A-Za-z0-9_ ]+".r
  val timestamp: Parser[String] = "2015-[0-9][0-9]-[0-9][0-9] [0-9:,]+".r
  val url: Parser[String] = "http://[a-zA-Z0-1.]+:[0-9]+/[a-zA-Z0-9_/]+".r
  val str: Parser[String] = "[A-Za-z0-9_:#.= ]+".r
  val before_event: Parser[String] = "^(?:(?!event).)*+".r
  val clean_apl: Parser[String] = "^(?:(?!_000001).)*+".r
}


object LogAnalyzer {
  def startParsing(sparkContext: SparkContext) = {
    
   // val sc = new SparkContext(new SparkConf().setAppName("Test"))
    val lines = sparkContext.textFile("hdfs:///datasets/clusterlogs/yarn-yarn-resourcemanager-master1.log*")

    def parseLine(l: String): LogLine =
      LogP.parse(LogP.logline, l).getOrElse(UnknownLine())

    def f(a: LogLine) = a match {
      case AppSummary(t, app, name, user, state, url, host, stime, etime, finalStatus) => List(app+" "+finalStatus) 
   
      case CapacitySchedulery(t, app, noname, event) => 
      {
        if (event == "RELEASED") {
          List(app+" "+event)
        } else {
          List()
        }
      }
     
      case _ => List()
    }

    val ll = lines.map(l => parseLine(l)).flatMap(f).cache
    ll.saveAsTextFile("app")  // in the user's HDFS home directory
    
  }
}

object Test {
  def main(args: Array[String]) {
    
    //Configures  
    val conf = new SparkConf().setAppName("Failure Prediction(home assignment 2)")
    val sc = new SparkContext(new SparkConf().setAppName("Test"))

    //Start parsing
    LogAnalyzer.startParsing(sc)

    // Load training data.
    val rawdata = sc.textFile("hdfs:///user/shiroche/app/part-*")

    //Parse data by two logs lines
    val data = rawdata.map( x=> (x.split(" ")(0), x.split(" ")(1)) )
    val released = data.filter{ case (key, value) => value == "RELEASED" }
    val statused = data.filter{ case (key, value) => value != "RELEASED" }

    //Merge logs lines with sum of CapacityScheduler
    val released_count = released.map( x => x._1).map(x => (x,1)).reduceByKey((x,y) => x + y)
    val work = statused.leftOuterJoin(released_count)   

    //Convert data for machine learing algoruthm
    val dataFinal = work.map(s => s._2).map{ x=> 
      if (x._1 == "FAILED") (1, x._2.getOrElse(0)) else (0, x._2.getOrElse(0))
    }

    // Convert to the LIBSVM format.
    val dataTrain = dataFinal.map( t => LabeledPoint(t._1.toDouble, Vectors.dense(t._2.toDouble))) 
   
    var i = 0
    var aver_auROC:Double = 0
    for( i <- 1 to 5){
        // Split data into training (60%) and test (40%).
        val splits = dataTrain.randomSplit(Array(0.6, 0.4), seed = 11L)
        val training = splits(0).cache()
        val test = splits(1)

        //
        val model = new LogisticRegressionWithLBFGS().run(training)
        
        //val numIterations = 100
        //val model = SVMWithSGD.train(training, numIterations)

        // Clear the default threshold.
       // model.clearThreshold()

        //Computing areaUnderROC for predictions.
        val scoreAndLabels = test.map { point =>
            val score = model.predict(point.features)
            (score, point.label)
        }
        val metrics = new BinaryClassificationMetrics(scoreAndLabels)
        val auROC = metrics.areaUnderROC()
        
        aver_auROC = aver_auROC + auROC    
        println(i+" time: Area under ROC =" + auROC)

    }
   
    //Print the average of 5 times calculation auROC
    aver_auROC = aver_auROC/5    
    println(" Average Area under ROC =" + aver_auROC)

    sc.stop()
  }

}
