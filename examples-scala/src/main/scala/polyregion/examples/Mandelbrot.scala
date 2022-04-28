package polyregion.examples

import polyregion.scala.{Buffer, NativeStruct}

import scala.reflect.ClassTag

object Mandelbrot {

  given NativeStruct[Colour] = polyregion.scala.compiletime.nativeStructOf

  given NativeStruct[Complex] = polyregion.scala.compiletime.nativeStructOf

  final val Palette = Array[Colour](
    Colour(66, 30, 15),
    Colour(25, 7, 26),
    Colour(9, 1, 47),
    Colour(4, 4, 73),
    Colour(0, 7, 100),
    Colour(12, 44, 138),
    Colour(24, 82, 177),
    Colour(57, 125, 209),
    Colour(134, 181, 229),
    Colour(211, 236, 248),
    Colour(241, 233, 191),
    Colour(248, 201, 95),
    Colour(255, 170, 0),
    Colour(204, 128, 0),
    Colour(153, 87, 0),
    Colour(106, 52, 3)
  )

  final val Palette2 = Buffer[Colour](Palette*)

  case class Complex(real: Double, imag: Double) {
    def inverse: Complex = {
      val denom = real * real + imag * imag
      Complex(real / denom, -imag / denom)
    }
    def +(b: Complex): Complex    = Complex(real + b.real, imag + b.imag)
    def -(b: Complex): Complex    = Complex(real - b.real, imag - b.imag)
    def *(b: Complex): Complex    = Complex(real * b.real - imag * b.imag, real * b.imag + imag * b.real)
    def /(b: Complex): Complex    = this * b.inverse
    def unary_- : Complex         = Complex(-real, -imag)
    def abs: Double               = math.hypot(real, imag)
    override def toString: String = s"$real + ${imag}i"
  }
  object Complex {
    val Zero: Complex = Complex(0.0, 0.0)
  }

  def interpolate(input: Double, inputMin: Double, inputMax: Double, outputMin: Double, outputMax: Double): Double =
    ((outputMax - outputMin) * (input - inputMin) / (inputMax - inputMin)) + outputMin

  case class Colour(packed: Int) {
    def a: Int = (packed >> 24) & 0xff
    def r: Int = (packed >> 16) & 0xff
    def g: Int = (packed >> 8) & 0xff
    def b: Int = packed & 0xff
    def mix(that: Colour, x: Double): Colour = Colour(
      r = ((r - that.r) * x + that.r).toInt,
      g = ((g - that.g) * x + that.g).toInt,
      b = ((b - that.b) * x + that.b).toInt,
      a = ((a - that.a) * x + that.a).toInt
    )
  }

  object Colour {
    final val Black: Colour             = Colour(0, 0, 0)
    final val White: Colour             = Colour(255, 255, 255)
    private def clampUInt8(x: Int): Int = x.max(0).min(255)
    def apply(r: Int, g: Int, b: Int, a: Int = 255): Colour = Colour(
      (clampUInt8(a) & 0xff) << 24     //
        | (clampUInt8(r) & 0xff) << 16 //
        | (clampUInt8(g) & 0xff) << 8  //
        | (clampUInt8(b) & 0xff)       //
    )
  }

  def itMandel(c: Complex, imax: Int, bailout: Int): (Complex, Int) = {
    var z = Complex.Zero
    var i = 0
    while (z.abs <= bailout && i < imax) {
      z = z * z + c
      i += 1
    }
    (z, i)
  }

  case class ItResult[A, B](c: B, i: A) {
    def read[C]: A = i
  }
  case class ItResultFA(c: Complex, i: Int)

  def itMandel2(c: Complex, imax: Int, bailout: Int): ItResult[Complex, Int] = {
    var z = Complex.Zero
    var i = 0
    while (z.abs <= bailout && i < imax) {
      z = z * z + c
      i += 1
    }
    ItResult(i, z)
  }

  def mkColour(z: Complex, iter: Int, maxIter: Int): Colour =
    if (iter >= maxIter) Colour.Black
    else {
      val logZn = math.log(z.abs) / 2
      val nu    = math.log(logZn / math.log(2)) / math.log(2)
      Palette(iter % Palette.length).mix(Palette((iter + 1) % Palette.length), nu)
    }

  def mkColour2(z: Complex, iter: Int, maxIter: Int): Colour =
    if (iter >= maxIter) Colour.Black
    else {
      val logZn = math.log(z.abs) / 2
      val nu    = math.log(logZn / math.log(2)) / math.log(2)
      Palette2(iter % 16).mix(Palette2((iter + 1) % 16), nu)
    }

  object In {
    def takeOne: Int = {
//      Colour.Black.packed

      Colour.Black.packed

      1
//      1+1
    }
  }
  // given NativeStruct[Complex] = polyregion.compiletime.nativeStructOf
  // given NativeStruct[ItResult] = polyregion.compiletime.nativeStructOf

  def run(buffer: Array[Array[Colour]], width: Int, height: Int, poiX: Double, poiY: Double, scale: Double): Unit = {
    val maxIter      = 500
    val zoom         = 1d / scale
    val (xMin, xMax) = (poiX - zoom, poiX + zoom)
    val (yMin, yMax) = (poiY - zoom, poiY + zoom)
    import scala.collection.parallel.CollectionConverters.*
//    val x = polyregion.scala.compiletime.offload(1 + 1)

//    println(s"Go! ${x} w = ${width} + ${height}")

    val image = Buffer.ofZeroed[Colour](width * height)

    object A {
      final val Const: Int                      = ???
      def getter: Int                           = ???
      def genGetter[GenTpe]: GenTpe             = ???
      def genGetter2[GenTpe1, GenTpe2]: GenTpe1 = ???

      def arg0(): Unit                              = ???
      def arg1(i: Int): Unit                        = ???
      def gen[A](i: Int): A                         = ???
      def genImplicit[A: ClassTag]: A               = ???
      def curry(a: Int)(b: Int): Unit               = ???
      def genCurry[A, B](a: Int, b: B)(c: B): Unit  = ???
      def curryA(a: Int): Int => String => Unit     = ???
      def curryFn(): (a: Int, b: Int) => () => Unit = ???
      val f: (Int, Int) => Unit                     = ???
    }

    object B {
      def apply(): Unit = ???
    }

    class C {
      final val Field: Int = ???

      def clsFn(): Unit = ???
    }

    def localFn0: Unit  = ???
    def localFn(): Unit = ???

    val localF: () => Unit = ???

    // ap @ Apply(fun : q.Term, args : q.Terms)
    //

    val c                      = new C
    val f1: (Int, Int) => Unit = ???

    // Exec(A, Nil, Var(V))

    val m = System.getProperty("a")
//    polyregion.scala.compiletime.showExpr{
//
//      A.curryA(1)
//      import A._
//      arg0()
//
//      c.clsFn()
//      localFn0
//      localFn()
//      localF()
//
//
//      // : Exec
//
//      c.Field
//      A.Const
//
//
//      B()
//      scala.math.max(1,2)
//      scala.math.max(x = 1,2)
//      A.getter
//      A.genGetter[String]
//      A.arg0()
//      A.arg1(1)
//      A.arg1.apply(1)
//      A.gen[String](2)
//      A.genImplicit[String]
//      A.curry(1)(2) // A.curry (1) >>> (2)
//      A.genCurry[Int, Int](1, 2)(3) // A.genCurry[String, Int] (1) >>> (2)
//      A.curryFn().apply(1, 2).apply()
//      A.curryFn()(1, 2) // A.curryFn: MethodType
//      A.f(1,2)    // A.f      : AppliedType
//      f1(1,2)    // A.f      : AppliedType
//
//      "".length
//      m.length
//      ()
//
//    }

    object Foo {
      var m: Int = 1
    }

    def takeIt(foo: Foo.type): Unit =
      foo.m += 1

    val a = 1

    polyregion.scala.compiletime.offload {

      val x = 1 + 1.toDouble

//      B()
//      c.clsFn()
//      A.getter
//      A.gen[Int](2)
//      A.genGetter2[Int, Short]
//      localFn0
//      localFn()
////      A.gen[Int](2)
//
////      A.curryA(1)
//      A.genCurry[Int, Int](1,2)(3)

//      new Complex(1,2)\
      val s    = 1.0f
      val m    = new ItResult[Int, Float](s, 2)
      val out1 = m.read[Int]
      val out2 = m.read[Float]

      val m2 = new ItResult[Float, Int](2, 1.0f)

//      ItResultFA(Complex(1.0, 1.0), 1)
//      ItResult(Complex.Zero, 1)

//      var y = 0
//      while (y < height) {
//        var x = 0
//        while (x < width) {
//          val c  = Complex(interpolate(x, 0, width, xMin, xMax), interpolate(y, 0, height, yMin, yMax))
//          val t  = itMandel2(c, maxIter, 4)
//          val cc = mkColour2(t.c, t.i, maxIter)
//          image(x + (y * width)) = cc
//          //          buffer(x)(y) = cc
//          x += 1
//        }
//        y += 1
//      }
    }

    image.grouped(width).map(_.toArray).toArray.transpose.copyToArray(buffer)

//    println(image.grouped(width).size)
//    image.grouped(width).map(_.toArray).toArray.transpose.zipWithIndex.foreach { case (xs, i) =>  xs.copyToArray(buffer(i))  }

    println("Stop")

//    for {
//      y <- (0 until height).par
//      x <- 0 until width
//    } {
//
////      val m2 = polyregion.scala.compiletime.offload {
////        val c = Complex(interpolate(x, 0, width, xMin, xMax), interpolate(y, 0, height, yMin, yMax))
////        val t = itMandel2(c, maxIter, 4)
////        if (t.i >= maxIter) Colour(0, 0, 0)
////        else {
////          val logZn = math.log(t.c.abs) / 2
////          val nu    = math.log(logZn / math.log(2)) / math.log(2)
////          Palette2(t.i % 16).mix(Palette2((t.i + 1) % 16), nu)
////        }
////      }
////      buffer(x)(y) = m2
//
//      val c         = Complex(interpolate(x, 0, width, xMin, xMax), interpolate(y, 0, height, yMin, yMax))
//      val (z, iter) = itMandel(c, maxIter, 4)
//      buffer(x)(y) = mkColour(z, iter, maxIter)
//    }
    println("Finish")

  }

  def showImage(image: java.awt.image.BufferedImage) = {
    import java.awt.{BorderLayout, Color}
    import javax.swing.*

    val frame: JFrame = new JFrame
    frame.setTitle("Mandelbrot")

    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    frame.getContentPane.setBackground(Color.BLACK)
    frame.getContentPane.setLayout(new BorderLayout)
    frame.getContentPane.add(new JLabel(new ImageIcon(image)))
    frame.pack()
    frame.setVisible(true)
  }

  def main(args: Array[String]): Unit = {
    import java.awt.image.BufferedImage

    println("Go")
    val image = new BufferedImage(128, 128, BufferedImage.TYPE_INT_ARGB)

    val data = Array.ofDim[Colour](image.getWidth, image.getHeight)

    val N       = 1
    val elapsed = Array.ofDim[Long](N)
    for (i <- 0 until N) {
      val start = System.nanoTime()
//      run(data, image.getWidth, image.getHeight, 0.28693186889504513, 0.014286693904085048, 10000 + i)
      run(data, image.getWidth, image.getHeight, -0.7, 0, 1.0)
      elapsed(i) = System.nanoTime() - start
    }

    println(elapsed.map(_.toDouble / 1e6).mkString(", "))

    for {
      x <- 0 until image.getWidth
      y <- 0 until image.getHeight
    } yield image.setRGB(x, y, data(x)(y).packed)
    println("Done")

    showImage(image)

  }

}
