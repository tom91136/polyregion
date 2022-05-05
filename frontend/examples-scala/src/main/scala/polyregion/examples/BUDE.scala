package polyregion.examples

import better.files.File

import java.nio.{ByteBuffer, ByteOrder}
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.compiletime.constValue

object BUDE {

  final inline val ZERO    = 0f
  final inline val QUARTER = 0.25f
  final inline val HALF    = 0.5f
  final inline val ONE     = 1f
  final inline val TWO     = 2f
  final inline val FOUR    = 4f
  final inline val CNSTNT  = 45f

  // Energy evaluation parameters
  final inline val HBTYPE_F = 70
  final inline val HBTYPE_E = 69
  final inline val HARDNESS = 38f
  final inline val NPNPDIST = 5.5f
  final inline val NPPDIST  = 1f

  case class Atom(x: Float, y: Float, z: Float, tpe: Int)
  case class FFParams(hbtype: Int, radius: Float, hphb: Float, elsc: Float)
  case class Vec3f(x: Float, y: Float, z: Float)
  case class Vec4f(x: Float, y: Float, z: Float, w: Float)

  case class RangeClosedInternal(from: Int, to: Int, step: Int) {
    inline def foreach(inline f: Int => Unit): Unit = {
      var i = from
      while (i < to) { f(i); i += step }
    }
  }

  extension (x: Int) {
    infix inline def until2(inline y: Int): RangeClosedInternal = RangeClosedInternal(x, y, 1)
    infix inline def to(inline y: Int): RangeClosedInternal     = RangeClosedInternal(x, y + 1, 1)
  }

  def fasten_main(
      proteins: Array[Atom],
      ligands: Array[Atom],
      forcefield: Array[FFParams],
      transforms: Array[Array[Float]],
      energies: Array[Float],
      PPWI: Int,
      group: Int
  ): Unit = {

    val transform = Array.ofDim[Vec4f](PPWI, 3)
    val etot      = Array.ofDim[Float](PPWI)

    for (l <- 0.until2(PPWI)) {
      val ix = group * PPWI + l
      val sx = math.sin(transforms(0)(ix)).toFloat
      val cx = math.cos(transforms(0)(ix)).toFloat
      val sy = math.sin(transforms(1)(ix)).toFloat
      val cy = math.cos(transforms(1)(ix)).toFloat
      val sz = math.sin(transforms(2)(ix)).toFloat
      val cz = math.cos(transforms(2)(ix)).toFloat
      transform(l)(0) = Vec4f(cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz, transforms(3)(ix))
      transform(l)(1) = Vec4f(cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz, transforms(4)(ix))
      transform(l)(2) = Vec4f( /**/ -sy, /*        */ sx * cy, /*          */ cx * cy, transforms(5)(ix))
    }

    for {
      l_atom <- ligands
      l_params  = forcefield(l_atom.tpe)
      lhphb_ltz = l_params.hphb < ZERO
      lhphb_gtz = l_params.hphb > ZERO
      lpos = transform.map { xform =>
        Vec3f(
          xform(0).w + l_atom.x * xform(0).x + l_atom.y * xform(0).y + l_atom.z * xform(0).z,
          xform(1).w + l_atom.x * xform(1).x + l_atom.y * xform(1).y + l_atom.z * xform(1).z,
          xform(2).w + l_atom.x * xform(2).x + l_atom.y * xform(2).y + l_atom.z * xform(2).z
        )
      }

      p_atom <- proteins
      p_params = forcefield(p_atom.tpe)

      radij   = p_params.radius + l_params.radius
      r_radij = ONE / radij

      elcdst  = if (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) FOUR else TWO
      elcdst1 = if (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) QUARTER else HALF
      type_E  = p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E

      phphb_ltz = p_params.hphb < ZERO
      phphb_gtz = p_params.hphb > ZERO
      phphb_nz  = p_params.hphb.abs != ZERO
      p_hphb    = p_params.hphb * (if (phphb_ltz && lhphb_gtz) -ONE else ONE)
      l_hphb    = l_params.hphb * (if (phphb_gtz && lhphb_ltz) -ONE else ONE)
      distdslv =
        if (phphb_ltz) {
          if (lhphb_ltz) NPNPDIST else NPPDIST
        } else {
          if (lhphb_ltz) NPPDIST else -Float.MaxValue
        }
      r_distdslv = ONE / distdslv
      chrg_init  = l_params.elsc * p_params.elsc
      dslv_init  = p_hphb + l_hphb

      l <- 0.until2(PPWI)
    } {
      // Calculate distance between atoms
      val x: Float      = lpos(l).x - p_atom.x
      val y: Float      = lpos(l).y - p_atom.y
      val z: Float      = lpos(l).z - p_atom.z
      val distij: Float = math.sqrt(x * x + y * y + z * z).toFloat

      // Calculate the sum of the sphere radii
      val distbb: Float = distij - radij

      val zone1: Boolean = distbb < ZERO

      //  Calculate steric energy
      etot(l) += (ONE - (distij * r_radij)) * (if (zone1) TWO * HARDNESS else ZERO)

      // Calculate formal and dipole charge interactions
      var chrg_e: Float =
        chrg_init * (if (zone1) ONE else ONE - distbb * elcdst1) * (if (distbb < elcdst) ONE else ZERO)
      val neg_chrg_e: Float = -math.abs(chrg_e)
      chrg_e = if (type_E) neg_chrg_e else chrg_e
      etot(l) += chrg_e * CNSTNT

      // Calculate the two cases for Nonpolar-Polar repulsive interactions
      val coeff: Float  = ONE - (distbb * r_distdslv)
      var dslv_e: Float = dslv_init * (if (distbb < distdslv && phphb_nz) ONE else ZERO)
      dslv_e *= (if (zone1) ONE else coeff)
      etot(l) += dslv_e
    }

    for (l <- 0.until2(PPWI)) energies(group * PPWI + l) = etot(l) * HALF

  }

  def readNStruct[A: ClassTag](file: File, structSizeInBytes: Int)(f: ByteBuffer => A): Array[A] = {
    val bytes = file.size
    require(bytes % structSizeInBytes == 0)
    val drain = ArrayBuffer[A]()
    file.fileInputStream { s =>
      while (s.available() > 0)
        drain += f(ByteBuffer.wrap(s.readNBytes(structSizeInBytes)).order(ByteOrder.LITTLE_ENDIAN))
    }
    drain.toArray
  }

  def main(args: Array[String]): Unit = {

    val pwd  = File.currentWorkingDirectory
    val deck = pwd / "src" / "main" / "resources" / "bude" / "bm1"

    def mkAtom(b: ByteBuffer) = Atom(x = b.getFloat, y = b.getFloat, z = b.getFloat, tpe = b.getInt)
    def mkFFParam(b: ByteBuffer) =
      FFParams(hbtype = b.getInt, radius = b.getFloat, hphb = b.getFloat, elsc = b.getFloat)

    val ligands     = readNStruct(deck / "ligand.in", 4 * 4)(mkAtom(_))
    val proteins    = readNStruct(deck / "protein.in", 4 * 4)(mkAtom(_))
    val forcefields = readNStruct(deck / "forcefield.in", 4 * 4)(mkFFParam(_))
    val poseFloats  = readNStruct(deck / "poses.in", 4)(_.getFloat)

    val poseCount = poseFloats.length / 6
    require(poseFloats.length % 6 == 0)
    val transformPoses = poseFloats.grouped(poseCount).toArray

    val refEnergies = (deck / "ref_energies.out").lines.map(_.toFloat).toArray
    require(refEnergies.length >= poseCount)

    println(s"""
		 |Ligands:     ${ligands.length}
         |Proteins:    ${proteins.length}
         |ForceFields: ${forcefields.length}
		 |Poses:       $poseCount
		 |Ref:         ${refEnergies.length}
         |""".stripMargin)

    val results = Array.ofDim[Float](poseCount)
    val PPWI    = 128
    val iters   = 20

    import scala.collection.parallel.CollectionConverters.*

    (poseCount % PPWI).ensuring(_ == 0)

    val elapsed = Array.ofDim[Long](iters)
    for (iter <- 0 until iters) {
      val start = System.nanoTime()
      (0 until (poseCount / PPWI)).par.map { group =>
        fasten_main(
          proteins = proteins,
          ligands = ligands,
          forcefield = forcefields,
          transforms = transformPoses,
          energies = results,
          PPWI = PPWI,
          group = group
        )
      }
      elapsed(iter) = System.nanoTime() - start
    }

    println(elapsed.map(_.toDouble / 1e6).mkString(", "))
    println(s"min=${elapsed.min.toDouble / 1e6}ms")

    results.take(10).zip(refEnergies.take(10)).foreach(println(_))

  }

}
