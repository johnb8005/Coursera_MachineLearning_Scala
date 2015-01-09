package models

import breeze.linalg._

// needed for universal functions such as exp
import breeze.math._
import breeze.numerics._

import breeze.optimize._
import breeze.stats._


object CourseraMachineLearning{	


	def KMeansExample{
		val X:DenseMatrix[Double] = utils.Math.readMAT("public/dataset/ex7data2.mat", "X")
		
		val initial_centroids:DenseMatrix[Double] = DenseMatrix((3.0, 3.0),(6.0,2.0),(8.0,5.0));
		val K = initial_centroids.rows; // 3 Centroids
		val idx = Kmean.findClosestCentroids(X, initial_centroids)

		val centroids = Kmean.computeCentroids(X, idx, K);

		println(centroids)


		// Find the closest centroids for the examples using the
		// initial_centroids
		//idx = findClosestCentroids(X, initial_centroids);

		//begin fx
	}
	def retrieveCoeffMatricesFromVector(vecCoeff: DenseVector[Double], s: Seq[Int]): Seq[DenseMatrix[Double]] = {
		var Theta:Seq[DenseMatrix[Double]] = Seq()
		var idx_s = 0
		for(i <- 1 until s.size){
			val idx_t = (s(i-1)+1)*s(i) + idx_s
			Theta :+= vecCoeff(idx_s to idx_t-1).toDenseMatrix.reshape(s(i), s(i-1)+1)
			idx_s = idx_t
		}
		Theta
	}

	// cost function "my" way
	def myCostFunction(
		nn_params: DenseVector[Double] // thetas, param to be optimizedSeq[DenseMatrix[Double]]
		, m: Int 							// number of observations
		, s: Seq[Int] 						// layer size (1: input, end: output, else: hidden layers)
		, X: DenseMatrix[Double] 		// input matrix
		, Y: DenseMatrix[Double] 		// output matrix/vector
		, lambda: Double = 0.0 			// regularization parameter
	): (Double, DenseVector[Double]) = {

		// retrieving Theta matrices from vector
		// change to function
		var Theta = retrieveCoeffMatricesFromVector(nn_params, s)


		//println("Number of thetas: "+Theta.size)
		
		// forward propagation
		var a:Seq[DenseMatrix[Double]] = Seq(X)
		for(i <- 0 until s.size-1){
			a :+= sigmoid(utils.Math.LinAlg.addOnes(a(i))*Theta(i).t)
		}

		//println("Number of as: "+a.size)

		// cost
		val cost = (Y:*log(a(a.size-1)):*(-1.0)) + ((Y:-1.0):*log((a(a.size-1):*(-1.0))+1.0))

		val t:Double = Theta.map{theta =>
			sum(pow(theta(::, 1 to -1),2))
		}.sum

		val L = lambda/((2*m).toDouble)*t

		val J = sum(cost)/m + L

		///// Backward propagation

		// grad must be of same dimension than theta
		// init
		var Theta_grad:Seq[DenseMatrix[Double]] = Theta.map{t =>
			DenseMatrix.zeros[Double](t.rows, t.cols)	
		}

		for(i <- 0 to m-1){

			// 1. perform forward pass
			var a:Seq[DenseVector[Double]] = Seq(X(i,::).t)
			for(i <- 0 until s.size-1){
				a :+=  sigmoid(Theta(i)*DenseVector.vertcat(DenseVector(1.0), a(i)))
			}

			// init delta
			var delta:Array[DenseVector[Double]] = new Array[DenseVector[Double]](s.size-1)

			// 2.
			delta(delta.size-1) = a(s.size-1) - Y(i, ::).t

			// 3.
			for(k <- s.size-3 to 0 by -1){
				delta(k) = {
					val t1 = Theta(k+1).t*delta(k+1)
					val t2 = (DenseVector.vertcat(DenseVector(1.0), utils.Math.LinAlg.sigmoidGradient(Theta(k)*DenseVector.vertcat(DenseVector(1.0), a(k)))))
					val d = t1:*t2
					d(1 to -1)
				}
			}

			for(k <- 0 to s.size-2){
				Theta_grad(k) :+= delta(k)*DenseVector.vertcat(DenseVector(1.0), a(k)).t
			}
		}

		var grad:DenseVector[Double] = DenseVector.zeros[Double](0) 

		// adding normalizations, regularization and packing into vector
		Theta_grad.zipWithIndex.map{case(t, i) =>
			t :/= m.toDouble
			t(::, 1 to -1) :+= Theta(i)(::, 1 to -1):*(lambda/m.toDouble)
			grad = DenseVector.vertcat(grad, t.toDenseVector)
		}

		(J,grad)
	}

	// cost function the coursera way
	def nnCostFunction(
		nn_params: DenseVector[Double] // thetas, param to be optimizedSeq[DenseMatrix[Double]]
		, m: Int 							// number of observations
		, n: Int 							// input layer size
		, s1: Int 							// hidden layer size
		, p: Int 							// number of outputs/labels
		, X: DenseMatrix[Double] 		// input matrix
		, Y: DenseMatrix[Double] 		// output matrix/vector
		, lambda: Double = 0.0 			// regularization parameter
	): (Double, DenseVector[Double]) = {

		val Theta1:DenseMatrix[Double] = nn_params(0 to ((n+1)*s1-1)).toDenseMatrix.reshape(s1, n+1)
		val Theta2:DenseMatrix[Double] = nn_params(((n+1)*s1) to -1).toDenseMatrix.reshape(p, s1+1)

		// forward propagation
		val a1 = utils.Math.LinAlg.addOnes(X)
		
		val z2 = a1*Theta1.t
		val a2 = sigmoid(z2)

		val z3 = utils.Math.LinAlg.addOnes(a2)*Theta2.t
		val a3 = sigmoid(z3)

		// cost
		val t = (Y:*log(a3):*(-1.0)) + ((Y:-1.0):*log((a3:*(-1.0))+1.0))

		val t1 = pow(Theta1(::, 1 to -1),2)
		val t2 = pow(Theta2(::, 1 to -1),2)
		val L = lambda/((2*m).toDouble)*(sum(t1)+sum(t2))

		val J = sum(t)/m + L

		///// Backward propagation

		// grad must be of same dimension than theta
		var Theta1_grad = DenseMatrix.zeros[Double](Theta1.rows, Theta1.cols)
		var Theta2_grad = DenseMatrix.zeros[Double](Theta2.rows, Theta2.cols)

		for(i <- 0 to m-1){

			// 1. perform forward pass
			val a1:DenseVector[Double] = DenseVector.vertcat(DenseVector(1.0), X(i,::).t)
			//println(a1.size)
			val z2 = Theta1*a1
			val a2 = DenseVector.vertcat(DenseVector(1.0), sigmoid(z2))
			val z3 = Theta2*a2
			val a3 = sigmoid(z3)

			// 2. 
			val delta3 = a3 - Y(i, ::).t

			val t1 = Theta2.t*delta3
			val t2 = (DenseVector.vertcat(DenseVector(1.0), utils.Math.LinAlg.sigmoidGradient(z2)))

			// 3.
			val delta2 = t1:*t2
			val delta22 = delta2(1 to -1)

			Theta1_grad :+= delta22*a1.t
			Theta2_grad :+= delta3*a2.t
		}

		Theta1_grad :/= m.toDouble
		Theta2_grad :/= m.toDouble

		// regularizing
		Theta1_grad(::, 1 to -1) :+= Theta1(::, 1 to -1):*(lambda/m.toDouble)
		Theta2_grad(::, 1 to -1) :+= Theta2(::, 1 to -1):*(lambda/m.toDouble)

		(J,DenseVector.vertcat(Theta1_grad.toDenseVector, Theta2_grad.toDenseVector))
	}



	// this is the neural network ex4 of Coursera
	def neural_network2{
		
		val filepath:String = "public/dataset/ex4data1.mat"
		val filepath2:String = "public/dataset/ex4weights.mat"

		// load data
		val X:DenseMatrix[Double] = utils.Math.readMAT(filepath, "X")
		val y:DenseMatrix[Double] = utils.Math.readMAT(filepath, "y")

		val n:Int = X.cols // number of parameters/features
		val m:Int = X.rows // number of observations | training data
		val p:Int = 10  // number of outputs/labels, y.rows

		// load pre-initiailsed weights (answer)
		val Theta1:DenseMatrix[Double] = utils.Math.readMAT(filepath2, "Theta1")
		val Theta2:DenseMatrix[Double] = utils.Math.readMAT(filepath2, "Theta2")

		// param optimization
		val s1:Int 			= 22 // 25 hidden units
		val S:Seq[Int] 	= Seq(n, s1, p)
		val lambda:Double 	= 1.0
		val mIterations:Int = 40

		// recoding y :: 
		var Y = DenseMatrix.zeros[Double](m, p)
		for(i <- 0 to m-1){
			Y(i, y(i,0).toInt-1) = 1.0
		}
		//println("Size of Y: "+Y.rows+" x "+Y.cols)



		// get initial parameters
		val initial_nn_params = {
			val size_vec:Int 	= {
				var r = 0
				
				for(i <- 0 to S.size-2){
					r += (S(i)+1)*S(i+1)
				}

				r
			}
			randInitializeWeight(size_vec)
		}
		println("size initial_nn_params "+initial_nn_params.size)
		println(sum(initial_nn_params))

		val r = myCostFunction(initial_nn_params, m, S, X, Y, lambda)

		println("After one iteration: ")
		println("cost J: "+r._1)
		println("grad: "+sum(r._2))

		def optimize ={
			val f = new DiffFunction[DenseVector[Double]] {
				def calculate(aV: DenseVector[Double]) = {
					myCostFunction(
						aV
						, m
						, S
						, X
						, Y
					)
					
				}
			}

			val lbfgs = new LBFGS[DenseVector[Double]](maxIter=mIterations, m=7)

			lbfgs.minimize(f,initial_nn_params)
		}

		
		println("== Optimization ==")
		val ntheta = optimize

		// here need to generalize!!
		// 1. create a function that retrieves theta from vector (take from myCostFunction)
		//val nTheta1 = ntheta(0 to ((n+1)*s1-1)).toDenseMatrix.reshape(s1, n+1)
		//val nTheta2 = ntheta(((n+1)*s1) to -1).toDenseMatrix.reshape(p, s1+1)

		val nTheta = retrieveCoeffMatricesFromVector(ntheta, S)



		// prediction
		var a:Seq[DenseMatrix[Double]] = Seq(X)
		for(i <- 0 until S.size-1){
			a :+= sigmoid(utils.Math.LinAlg.addOnes(a(i))*nTheta(i).t)
		}

		var b = DenseVector.zeros[Double](y.rows)

		for (i <- 0 to a.last.rows-1){
			var d:Double = 0.0
			val t:DenseVector[Double] = a.last(i, ::).t.toDenseVector
			for(j <- 0 to 9){
				if(t(j)>d){
					d = t(j)
					b(i) = (j+1)
				}
			}
		}

		val errors = (b - y.toDenseVector).map{a =>
			if(a==0){
				1
			}
			else{
				0
			}
		}

		//print(b)
		println(sum(errors).toDouble/m.toDouble)
		
		
	}

	def randInitializeWeight(L_in: Int, L_out: Int): DenseMatrix[Double] = {
		val epsilon:Double = 0.12
		DenseMatrix.rand(L_out, L_in + 1):*2.0:*epsilon - epsilon
	}

	def randInitializeWeight(L: Int): DenseVector[Double] = {
		val epsilon:Double = 0.12
		DenseVector.rand(L)*2.0*epsilon - epsilon
	}


	// this is the neural network ex3 of Coursera
	def neural_network{
		
		val filepath:String = "public/dataset/ex3data1.mat"
		val filepath2:String = "public/dataset/ex3weights.mat"

		// load
		val X:DenseMatrix[Double] = utils.Math.readMAT(filepath, "X")
		val y:DenseMatrix[Double] = utils.Math.readMAT(filepath, "y")

		// load pre-initiailsed weights (answer)
		val Theta1:DenseMatrix[Double] = utils.Math.readMAT(filepath2, "Theta1")
		val Theta2:DenseMatrix[Double] = utils.Math.readMAT(filepath2, "Theta2")

		val n:Int = X.cols // number of parameters/features
		val m:Int = X.rows // number of observations | training data
		println("Size of input Matrix X: "+m+" x "+n)
		println("Size of y: "+y.rows+" x "+y.cols)


		//val p:Int = 10  // number of outputs/labels
		//val hidden_layer_size:Int = 25 // 25 hidden units

		println("Size of theta1: "+Theta1.rows+" x "+Theta1.cols)
		println("Size of theta2: "+Theta2.rows+" x "+Theta2.cols)

		val a1 = sigmoid(utils.Math.LinAlg.addOnes(X)*Theta1.t)
		val a2 = sigmoid(utils.Math.LinAlg.addOnes(a1)*Theta2.t)

		println("Size of a2: "+a2.rows+" x "+a2.cols)

		var b = DenseVector.zeros[Double](y.rows)
		//println("Size of b: "+b.rows+" x "+b.cols)

		for (i <- 0 to a2.rows-1){
			//println(sum(a2(i, ::).t.toDenseVector))

			var d:Double = 0.0
			val t:DenseVector[Double] = a2(i, ::).t.toDenseVector
			for(j <- 0 to 9){
				if(t(j)>d){
					d = t(j)
					b(i) = (j+1)
				}
			}
		}


		val errors = (b - y.toDenseVector).map{a =>
			if(a==0){
				1
			}
			else{
				0
			}
		}
		//print(b)
		println(sum(errors).toDouble/5000.0)

	}
	


	/*
		Solving a multivar linear regression
		status: fininshed both with normal and gradient descent
	*/
	def linear_example{
		import models.Machine_learning._

		val y:breeze.linalg.DenseMatrix[Double]
			= DenseMatrix(1.0,2.0,3.0,7.0,8.0,6.0,9.0)

		var X:breeze.linalg.DenseMatrix[Double] 
			= {DenseMatrix(
				(1.0,2.0,3.0,4.0,5.0,6.0,7.0)
				,(2.0,3.0,4.0,2.0,3.0,7.0,6.0)
			)}.t

		val r = new Regression_linear(X, y)

		println("X")
		//println(X.t.toString)
		println("y")
		//println(y.toString)

		println("Solving linear regression using Gradient descent")
		println(r.solve_gradient_descent)
		println("Solving linear regression using Normal method")
		println(r.solve_analytical)

		println("Errors: "+r.errorsInSample.toString)
	}


	/* 
	this example is taken from ex2 of the Machine Learning course of Coursera
	- reads CSV file and creates two matrices X/y
	- creates the log regression object
	- does the optimisation
	- gives a probability of outcome for a sample
	- predicts for a sample
	- measure of accuracy included too
	*/
	def logistic_example {
		import models.Machine_learning._

		val Z = utils.Math.readCSV("public/dataset/testdata1.txt")

		val X:DenseMatrix[Double] = Z(::,0 to 1)
		val y:DenseMatrix[Double] = Z(::,2).toDenseMatrix

		val r = new Regression_logistic(X, y)

		r.gradient_cost

		//println("Solving linear regression using Log method")
		//println(r.solve_gradient_descent)

		//println("Errors: "+r.errorsInSample.toString)
		val theta = DenseMatrix(r.optimize.toArray).t

		println("theta")
		println(theta)

		//println(r.probability(utils.Math.LinAlg.addOnes(DenseMatrix((45.0,85.0))), theta))
		val p = r.predict(utils.Math.LinAlg.addOnes(X), theta).t
		println("predict")
		println(mean((y-p).map{a => if(a==0){1.0}else{0.0}}))
	}

	/*
	this is the same as logistic_example but includes regularization
	- inputs are mapped higer ordder polynomials
	- use of regularization to only keept relevant parameters
	*/
	def logistic_example_w_regularization {
		import models.Machine_learning._

		val Z = utils.Math.readCSV("public/dataset/testdata2.txt")

		val X:DenseMatrix[Double] = mapFeature(Z(::,0 to 1))
		val y:DenseMatrix[Double] = Z(::,2).toDenseMatrix

		val lambda = Some(1.0)

		val r = new Regression_logistic(X, y, lambda)

		r.gradient_cost

		//println("Solving linear regression using Log method")
		//println(r.solve_gradient_descent)

		//println("Errors: "+r.errorsInSample.toString)
		val theta = DenseMatrix(r.optimize.toArray).t

		println("theta")
		println(theta)

		//println(r.probability(utils.Math.LinAlg.addOnes(DenseMatrix((45.0,85.0))), theta))
		val p = r.predict(utils.Math.LinAlg.addOnes(X), theta).t
		println("predict")
		println(mean((y-p).map{a => if(a==0){1.0}else{0.0}}))
	}

	/*
		textbook example of the optimization of a simple function - taken from the doc
	*/
	def optim_essai{
		val f = new DiffFunction[DenseVector[Double]] {
			def calculate(x: DenseVector[Double]) = {
				(norm(((x+8.0) - 3.0 ) :^ 2.0,1.0),(x * 2.0) - 6.0)
			}
		}

		val lbfgs = new LBFGS[DenseVector[Double]](maxIter=100, m=3) // m is the memory. anywhere between 3 and 7 is fine. The larger m, the more memory is needed.
		val res 	= lbfgs.minimize(f,DenseVector(0,0,0))
		println(res)
		println(f(res))

		println("do the optimization")
		println(f.valueAt(DenseVector(0,0,3)))
		println(f.gradientAt(DenseVector(0,0,3)))
		println(f.calculate(DenseVector(0,0)))
	}

}
