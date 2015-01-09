package models

import breeze.linalg._

// needed for universal functions such as exp
import breeze.math._
import breeze.numerics._

import breeze.optimize._
import breeze.stats._



/*
	supervised learning
*/
object Machine_learning{


	/*
		generates polynomical features (x, x^2, x^3, etc)
	*/
	def mapFeature(X: DenseMatrix[Double]):DenseMatrix[Double] = {
		val degree:Int = 6

		var out:DenseMatrix[Double] = DenseMatrix.zeros[Double](X(::,1).size,1)

		var yu:Seq[DenseMatrix[Double]] = Seq()
		for (i <- 1 to degree){
			for (j <- 0 to i){
				val a:DenseVector[Double] = (pow(X(::, 0),i-j):*pow(X(::,1),j))
				val b:DenseMatrix[Double] = DenseMatrix(a.toArray).t
				out = DenseMatrix.horzcat(out,b)
			}
		}

		//remove the first column
		out(::,1 to -1)
	}


	class Regression_linear(Xc:breeze.linalg.DenseMatrix[Double], yc:breeze.linalg.DenseMatrix[Double]){

		val X 							= utils.Math.LinAlg.addOnes(Xc)
		val y 							= yc
		// attributes
		val n:Int 						= X.cols
		val m:Int 						= X.rows

		val alpha:Double 				= .01
		val N:Int 						= 9000
		val lambda:Option[Double] 	= None

		// initialize "a" for gradient descent
		var a:breeze.linalg.DenseMatrix[Double] = null

		lazy val solve_analytical 	= (X.t * X)\X.t*y

		def gradient_cost: breeze.linalg.DenseMatrix[Double] = {
			val h 	= X*a
			val d 	= h - y
			val k 	= X.t*d

			k:*(1/{m.toDouble})
		}

		def solve_gradient_descent: breeze.linalg.DenseMatrix[Double] = {
			var i:Int = 1
			a = DenseMatrix.zeros[Double](n,1)
			for (i <- 1 to N){
				a -= gradient_cost:*alpha
			}
			a
		}

		lazy val predictInSample 	= predict(X)
		lazy val errorsInSample 	= errors(X,y)

		def predict(X:breeze.linalg.DenseMatrix[Double]) 
											= X*a
		def errors(X:breeze.linalg.DenseMatrix[Double], y:breeze.linalg.DenseMatrix[Double]):Double
											= sum((predict(X) - y):^2.0)
	}

	class Regression_logistic(
		  Xc:breeze.linalg.DenseMatrix[Double]
		, yc:breeze.linalg.DenseMatrix[Double]
		, lambdac: Option[Double] = None
		, thresholdc: Option[Double] = None
		, iterationsc : Option[Int] = None
	){

		// add a column of ones
		val X:breeze.linalg.DenseMatrix[Double] 							= utils.Math.LinAlg.addOnes(Xc)
		val y:breeze.linalg.DenseMatrix[Double] 							= yc
		// attributes
		val n:Int 						= X.cols
		val m:Int 						= X.rows

		// param gradient descent
		val alpha:Double 				= .03
		val N:Int 						= 10000

		// param optim LFBGS
		val mIterations = {if(iterationsc.isDefined){iterationsc.get}else{300}}

		// param regularization
		val lambda:Option[Double] 	= lambdac
		// param threshold to accept sample or not
		val threshold:Double 		= {if(thresholdc.isDefined){thresholdc.get}else{.5}}

		// Coefficient theta - this is the output: X theta = y
		// initialize "a" for gradient descent
		var a:breeze.linalg.DenseMatrix[Double] = DenseMatrix.zeros[Double](n,1)

		// computes J (cost function) and associated gradient (only returns grad)
		def gradient_cost: breeze.linalg.DenseMatrix[Double] = {
			val h 	= sigmoid(X*a)
			val d 	= h - y
			val k 	= X.t*d

			var ar 	= a
			ar(0,0) 	= 0.0


			var J   = ((-y*log(h)) - (-y:+1.0)*log(-h:+1.0)):*(1/{m.toDouble})

			if(lambda.isDefined){
				J += (ar.t*ar):*lambda.get/(2*m)
			}


			var grad = k:*(1/{m.toDouble})

			if(lambda.isDefined){
				grad += ar:*lambda.get/m
			}

			/*println("h: ")
			println(h)
			println("J: ")
			println(J)
			println("grad: ")
			println(grad)*/

			grad

			
		}

		def solve_gradient_descent: breeze.linalg.DenseMatrix[Double] = {
			var i:Int = 1

			a = DenseMatrix.zeros[Double](n,1)
			for (i <- 1 to N){
				a -= gradient_cost:*alpha
			}

			a
		}


		def optimize ={
			val f = new DiffFunction[DenseVector[Double]] {
				def calculate(aV: DenseVector[Double]) = {

					// need to transform vector to matrix
					val a: DenseMatrix[Double] = aV.toDenseMatrix.t

					val h 	= sigmoid(X*a) 
					val d 	= h - y
					
					
					var J  	= ((-y*log(h)) - (-y:+1.0)*log(-h:+1.0)):*(1/{m.toDouble})					
					val grad	= X.t*d:*(1/{m.toDouble})


					if(lambda.isDefined){
						// for reg
						var ar 	= a
						ar(0,0) 	= 0.0
						J += sqrt(ar):*(lambda.get/(2*{m.toDouble}))
						grad += ar:*lambda.get/{m.toDouble}
					}


					(
						J(0,0)
						,
						// need to transform matrix back into vector
						grad.toDenseVector
					)
				}
			}

			val lbfgs = new LBFGS[DenseVector[Double]](maxIter=mIterations, m=3)
			
			lbfgs.minimize(f,DenseVector.zeros[Double](n))
		}
		
		lazy val predictInSample 	= predict(X)
		lazy val errorsInSample 	= errors(X,y)

		def probability(X:breeze.linalg.DenseMatrix[Double], a:breeze.linalg.DenseMatrix[Double] = a) 
											= sigmoid(X*a)
		def predict(X:breeze.linalg.DenseMatrix[Double], a:breeze.linalg.DenseMatrix[Double] = a) 
											= sigmoid(X*a).map{c => if(c>= threshold){1.0}else{0.0}}
		def errors(X:breeze.linalg.DenseMatrix[Double], y:breeze.linalg.DenseMatrix[Double]):Double
											= sum((predict(X) - y):^2.0)
	}

}


/*
	unsupervised learning
*/
object Kmean{

	def computeCentroids(X: DenseMatrix[Double], idx:DenseVector[Int], K:Int): DenseMatrix[Double] = {
		var centroids = DenseMatrix.zeros[Double](K, X.cols)
		var c_mean = DenseVector.zeros[Double](K)

		var i:Int=0
		while(i < X.rows){
			centroids(idx(i),::) += X(i,::)
			c_mean(idx(i)) += 1

			i += 1
		}

		i = 0
		while(i < K){
			centroids(i,::) /= c_mean(i)

			i += 1
		}

		centroids
	}

	def findClosestCentroids(X: DenseMatrix[Double], centroids: DenseMatrix[Double]):DenseVector[Int] = {
		var idx = DenseVector.zeros[Int](X.rows)

		var a:Int=0
		while(a < X.rows){
	
			var z:Double=Double.MaxValue;


			var i:Int = 0
			var id:Int = 1
			while(i < centroids.rows){
				val y:Double = sqrt(sum((X(a, ::).t - centroids(i, ::).t):^2.0))
				if(y < z){
					z = y
					id = i
				}

				idx(a) = id
				i += 1
			}
			a += 1
		}
		idx
	}
}