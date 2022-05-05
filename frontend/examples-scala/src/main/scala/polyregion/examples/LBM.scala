package polyregion.examples

import better.files.File

import scala.reflect.ClassTag

object LBM {

  type Num = Float

  class Speed(
      var s0: Num = 0f,
      var s1: Num = 0f,
      var s2: Num = 0f,
      var s3: Num = 0f,
      var s4: Num = 0f,
      var s5: Num = 0f,
      var s6: Num = 0f,
      var s7: Num = 0f,
      var s8: Num = 0f
  )

  def parseInt(s: String): Either[Exception, Int] = s.trim.toIntOption.toRight(new Exception(s"No parse[Int]: $s"))
  def parseFloat(s: String): Either[Exception, Float] =
    s.trim.toFloatOption.toRight(new Exception(s"No parse[Float]: $s"))

  case class Param(nx: Int, ny: Int, maxIters: Int, reynoldsDim: Int, density: Num, accel: Num, omega: Num)
  object Param {
    def apply(file: File): Either[Exception, Param] =
      file.lines.toList match {
        case nxLn :: nyLn :: maxItersLn :: reynoldsDimLn :: densityLn :: accelLn :: omegaLn :: Nil =>
          for {
            nx          <- parseInt(nxLn)
            ny          <- parseInt(nyLn)
            maxIters    <- parseInt(maxItersLn)
            reynoldsDim <- parseInt(reynoldsDimLn)
            density     <- parseFloat(densityLn)
            accel       <- parseFloat(accelLn)
            omega       <- parseFloat(omegaLn)
          } yield Param(nx, ny, maxIters, reynoldsDim, density, accel, omega)
        case xs => Left(new Exception(s"No parse: $xs"))
      }
  }

  case class Simulation(param: Param, obstacles: Image2d[Boolean], occupied: Int)
  object Simulation {

    def apply(param: File, obstacle: File): Either[Exception, Simulation] =
      for {
        p <- Param(param)
        (error, blocks) = obstacle.lines.toList
          .map(_.trim)
          .partitionMap(_.split(" ").toList match {
            case xv :: yv :: "1" :: Nil =>
              for {
                x <- parseInt(xv)
                y <- parseInt(yv)
              } yield x -> y
            case xs => Left(new Exception(s"Bad obstacle line: $xs"))
          })

        _ <- error match {
          case Nil => Right(())
          case xs  => Left(new Exception(xs.map(_.getMessage).mkString("\n")))
        }
        obstacles = new Image2d[Boolean](p.nx, p.ny, false)
        _         = blocks.foreach { case (x, y) => obstacles.set(x, y)(true) }
      } yield Simulation(p, obstacles, blocks.size)
  }

  class Image2d[@specialized A: ClassTag](x: Int, y: Int, v: => A) {
    val data: Array[Array[A]]                                = Array.tabulate(x, y)((_, _) => v)
    inline def apply(inline x: Int, inline y: Int): A        = data(x)(y)
    inline def set(inline x: Int, inline y: Int)(a: A): Unit = data(x)(y) = a
    inline def foreach(inline f: (Int, Int, A) => Unit): Unit = {
      var y = 0
      while (y < this.y) {
        var x = 0
        while (x < this.x) {
          f(x, y, apply(x, y))
          x += 1
        }
        y += 1
      }
    }
  }

  def initialiseBuffer(param: Param): Image2d[Speed] = {
    val image   = new Image2d[Speed](param.nx, param.ny, new Speed())
    val w0: Num = param.density * 4f / 9f
    val w1: Num = param.density / 9f
    val w2: Num = param.density / 36f
    image.foreach { (_, _, s) =>
      s.s0 = w0
      s.s1 = w1
      s.s2 = w1
      s.s3 = w1
      s.s4 = w1
      s.s5 = w2
      s.s6 = w2
      s.s7 = w2
      s.s8 = w2
    }
    image
  }

  def avVelocity(sim: Simulation, image: Image2d[Speed]): Num = {
    var uSum: Num = 0f
    image.foreach { (x, y, s) =>
      if (!sim.obstacles(x, y)) {
        val localDensity = s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7 + s.s8
        val u_x          = (s.s1 + s.s5 + s.s8 - (s.s3 + s.s6 + s.s7)) / localDensity
        val u_y          = (s.s2 + s.s5 + s.s6 - (s.s4 + s.s7 + s.s8)) / localDensity
        uSum += math.sqrt((u_x * u_x) + (u_y * u_y)).toFloat
      }
    }
    uSum / (sim.param.nx * sim.param.ny - sim.occupied)
  }

  def calcReynolds(sim: Simulation, image: Image2d[Speed]): Num =
    avVelocity(sim, image) * sim.param.reynoldsDim / (1f / 6f * (2f / sim.param.omega - 1f))

  def write(sim: Simulation, image: Image2d[Speed], out: File): Unit = {
    out.clear()
    val c_sq = 1f / 3f
    image.foreach { (x, y, s) =>
      val (u_x, u_y, u, pressure) =
        if (sim.obstacles(x, y)) { // an occupied cell
          (0f, 0f, 0f, sim.param.density * c_sq)
        } else { // no obstacle
          val localDensity = s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7 + s.s8
          val u_x          = (s.s1 + s.s5 + s.s8 - (s.s3 + s.s6 + s.s7)) / localDensity
          val u_y          = (s.s2 + s.s5 + s.s6 - (s.s4 + s.s7 + s.s8)) / localDensity
          (u_x, u_y, math.sqrt((u_x * u_x) + (u_y * u_y)).toFloat, localDensity * c_sq)
        }
      out.appendLine(
        String.format(
          "%d %d %.12E %.12E %.12E %.12E %d\n",
          x,
          y,
          u_x,
          u_y,
          u,
          pressure,
          if (sim.obstacles(x, y)) 1 else 0
        )
      )
    }
  }

  def accelerateFlow(sim: Simulation, image: Image2d[Speed]): Unit = {
    val param = sim.param
    val w1    = param.density * param.accel / 9f
    val w2    = param.density * param.accel / 36f
    val y     = param.ny - 2
    (0 until param.nx)
      .foreach(x =>
        if (
          !sim.obstacles(x, y)
          && (image(x, y).s3 - w1) > 0f
          && (image(x, y).s6 - w2) > 0f
          && (image(x, y).s7 - w2) > 0f
        ) {
          image(x, y).s1 += w1
          image(x, y).s5 += w2
          image(x, y).s8 += w2
          image(x, y).s3 -= w1
          image(x, y).s6 -= w2
          image(x, y).s7 -= w2
        }
      )
  }

  def collision(sim: Simulation, cells: Image2d[Speed], buffer: Image2d[Speed]): Num = {

    val cSq      = 1f / 3f
    val x2cSq    = 2f * cSq
    val x2cSqcSq = 2f * cSq * cSq
    val w0       = 4f / 9f
    val w1       = 1f / 9f
    val w2       = 1f / 36f

    val param = sim.param

    var uSum = 0f

    cells.foreach { (x, y, _) =>
      val obs = sim.obstacles(x, y)

      val xR    = (x + 1) % param.nx
      val yR    = (y + 1) % param.ny
      val xL    = if (x == 0) param.nx - 1 else x - 1
      val yL    = if (y == 0) param.ny - 1 else y - 1
      val spd_0 = cells(x, y).s0
      val spd_1 = cells(xL, y).s1
      val spd_2 = cells(x, yL).s2
      val spd_3 = cells(xR, y).s3
      val spd_4 = cells(x, yR).s4
      val spd_5 = cells(xL, yL).s5
      val spd_6 = cells(xR, yL).s6
      val spd_7 = cells(xR, yR).s7
      val spd_8 = cells(xL, yR).s8

      val density = spd_1 + spd_5 + spd_8 +
        (spd_3 + spd_6 + spd_7) +
        (spd_0 + spd_2 + spd_4)
      val uX = (spd_1 + spd_5 + spd_8 - (spd_3 + spd_6 + spd_7)) / density
      val uY = (spd_2 + spd_5 + spd_6 -
        (spd_4 + spd_7 + spd_8)) / density
      val uSq = uX * uX + uY * uY

      val u1 = uX
      val u2 = uY
      val u3 = -uX
      val u4 = -uY
      val u5 = uX + uY
      val u6 = -uX + uY
      val u7 = -uX - uY
      val u8 = uX - uY

      val uSqOverX2cSq = uSq / x2cSq
      val w0Density    = w0 * density
      val w1Density    = w1 * density
      val w2Density    = w2 * density

      val dEqu0 = w0Density * (1f - uSqOverX2cSq)
      val dEqu1 = w1Density * (1f + u1 / cSq + (u1 * u1) / x2cSqcSq - uSqOverX2cSq)
      val dEqu2 = w1Density * (1f + u2 / cSq + (u2 * u2) / x2cSqcSq - uSqOverX2cSq)
      val dEqu3 = w1Density * (1f + u3 / cSq + (u3 * u3) / x2cSqcSq - uSqOverX2cSq)
      val dEqu4 = w1Density * (1f + u4 / cSq + (u4 * u4) / x2cSqcSq - uSqOverX2cSq)
      val dEqu5 = w2Density * (1f + u5 / cSq + (u5 * u5) / x2cSqcSq - uSqOverX2cSq)
      val dEqu6 = w2Density * (1f + u6 / cSq + (u6 * u6) / x2cSqcSq - uSqOverX2cSq)
      val dEqu7 = w2Density * (1f + u7 / cSq + (u7 * u7) / x2cSqcSq - uSqOverX2cSq)
      val dEqu8 = w2Density * (1f + u8 / cSq + (u8 * u8) / x2cSqcSq - uSqOverX2cSq)

      if (!obs) buffer(x, y).s0 = spd_0 + param.omega * (dEqu0 - spd_0)
      buffer(x, y).s1 = if (obs) spd_3 else spd_1 + param.omega * (dEqu1 - spd_1)
      buffer(x, y).s2 = if (obs) spd_4 else spd_2 + param.omega * (dEqu2 - spd_2)
      buffer(x, y).s3 = if (obs) spd_1 else spd_3 + param.omega * (dEqu3 - spd_3)
      buffer(x, y).s4 = if (obs) spd_2 else spd_4 + param.omega * (dEqu4 - spd_4)
      buffer(x, y).s5 = if (obs) spd_7 else spd_5 + param.omega * (dEqu5 - spd_5)
      buffer(x, y).s6 = if (obs) spd_8 else spd_6 + param.omega * (dEqu6 - spd_6)
      buffer(x, y).s7 = if (obs) spd_5 else spd_7 + param.omega * (dEqu7 - spd_7)
      buffer(x, y).s8 = if (obs) spd_6 else spd_8 + param.omega * (dEqu8 - spd_8)
      uSum += (if (!obs) math.sqrt(uSq).toFloat else 0f)
    }
    uSum / (param.nx * param.ny - sim.occupied)
  }

  def timestep(sim: Simulation, image: Image2d[Speed]): Unit = {
    val buffer = new Image2d[Speed](sim.param.nx, sim.param.ny, new Speed())
    val avVels = Array.ofDim[Num](sim.param.maxIters)
    for (iter <- 0 until sim.param.maxIters by 2) {
      accelerateFlow(sim, image)
      avVels(iter) = collision(sim, image, buffer)
      accelerateFlow(sim, buffer)
      avVels(iter + 1) = collision(sim, buffer, image)
    }
  }

  def main(args: Array[String]): Unit =
    for (_ <- 0 to 10) {

      val pwd   = File.currentWorkingDirectory
      val input = pwd / "src" / "main" / "resources" / "lbm"

      val sim = Simulation(input / "input_128x128.params", input / "obstacles_128x128.dat").fold(e => throw e, identity)
      println(s"sim = ${sim.param}")

      val buffer = initialiseBuffer(sim.param)

      val (elapsedNs, _) = time {
        timestep(sim, buffer)
      }

      println(s"Elapsed time: ${elapsedNs.toDouble / 1e6} ms")
      println(f"Reynolds number:${calcReynolds(sim, buffer)}%.12E")
//      write(sim, buffer, out)
    }

}
