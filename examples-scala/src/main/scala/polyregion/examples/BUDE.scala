package polyregion.examples

object BUDE {

  case class Atom(x: Float, y: Float, z: Float, tpe: Int)
  case class FFParams(hbtype: Int, radius: Float, hphb: Float, elsc: Float)

  case class Vec3[A](x: A, y: A, z: A)
  case class Vec4[A](x: A, y: A, z: A, w: A)

  val transforms: Array[Array[Float]] = ???

  val protein: Array[Atom]        = ???
  val ligand: Array[Atom]         = ???
  val forcefield: Array[FFParams] = ???

  val PPWI: Int = 4

  val ZERO    = 0.0f
  val QUARTER = 0.25f
  val HALF    = 0.5f
  val ONE     = 1.0f
  val TWO     = 2.0f
  val FOUR    = 4.0f
  val CNSTNT  = 45.0f

  // Energy evaluation parameters
  val HBTYPE_F = 70
  val HBTYPE_E = 69
  val HARDNESS = 38.0f
  val NPNPDIST = 5.5f
  val NPPDIST  = 1.0f

  def doIt = {

    val group: Int = ???
    val transform  = Array.ofDim[Vec4[Float]](3, PPWI)
    val etot       = Array.ofDim[Float](PPWI)

    for (l <- 0 until PPWI) {
      val ix = group * PPWI + l
      val sx = math.sin(transforms(0)(ix)).toFloat
      val cx = math.cos(transforms(0)(ix)).toFloat
      val sy = math.sin(transforms(1)(ix)).toFloat
      val cy = math.cos(transforms(1)(ix)).toFloat
      val sz = math.sin(transforms(2)(ix)).toFloat
      val cz = math.cos(transforms(2)(ix)).toFloat

      transform(0)(l) = Vec4(
        cy * cz,
        sx * sy * cz - cx * sz,
        cx * sy * cz + sx * sz,
        transforms(3)(ix)
      )
      transform(1)(l) = Vec4(
        cy * sz,
        sx * sy * sz + cx * cz,
        cx * sy * sz - sx * cz,
        transforms(4)(ix)
      )
      transform(2)(l) = Vec4(
        -sy,
        sx * cy,
        cx * cy,
        transforms(5)(ix)
      )
    }

    for {
      l_atom <- ligand
      l_params  = forcefield(l_atom.tpe)
      lhphb_ltz = l_params.hphb < 0
      lhphb_gtz = l_params.hphb > 0
      lpos = transform.map { xform =>
        Vec3(
          xform(0).w + //
            l_atom.x * xform(0).x +
            l_atom.y * xform(0).y +
            l_atom.z * xform(0).z,
          xform(1).w + //
            l_atom.x * xform(1).x +
            l_atom.y * xform(1).y +
            l_atom.z * xform(1).z,
          xform(2).w + //
            l_atom.x * xform(2).x +
            l_atom.y * xform(2).y +
            l_atom.z * xform(2).z
        )
      }

      p_atom <- protein
      p_params = forcefield(p_atom.tpe)

      radij   = p_params.radius + l_params.radius
      r_radij = ONE / radij

      elcdst  = if (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) FOUR else TWO
      elcdst1 = if (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F) QUARTER else HALF
      type_E  = p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E

      phphb_ltz = p_params.hphb < 0f
      phphb_gtz = p_params.hphb > 0f
      phphb_nz  = p_params.hphb != 0f
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

      l <- 0 until PPWI

    } {
      // Calculate distance between atoms
      val x      = lpos(l).x - p_atom.x
      val y      = lpos(l).y - p_atom.y
      val z      = lpos(l).z - p_atom.z
      val distij = math.sqrt(x * x + y * y + z * z).toFloat

      // Calculate the sum of the sphere radii
      val distbb = distij - radij

      val zone1 = distbb < ZERO

      //  Calculate steric energy
      etot(l) += (ONE - (distij * r_radij)) * (if (zone1) TWO * HARDNESS else ZERO)

      // Calculate formal and dipole charge interactions
      var chrg_e     = chrg_init * (if (zone1) ONE else ONE - distbb * elcdst1) * (if (distbb < elcdst) ONE else ZERO)
      val neg_chrg_e = math.abs(chrg_e)
      chrg_e = if (type_E) neg_chrg_e else chrg_e
      etot(l) += chrg_e * CNSTNT

      // Calculate the two cases for Nonpolar-Polar repulsive interactions
      val coeff  = ONE - (distbb * r_distdslv)
      var dslv_e = dslv_init * (if (distbb < distdslv && phphb_nz) ONE else ZERO)
      dslv_e *= (if (zone1) ONE else coeff)
      etot(l) += dslv_e;
    }
    for (l <- 0 until PPWI) etot(l) *= HALF

  }

}
