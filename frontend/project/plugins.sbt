addSbtPlugin("com.timushev.sbt" % "sbt-updates" % "0.7.0")
// addSbtPlugin("org.typelevel"             % "sbt-tpolecat"     % "0.5.2")
addSbtPlugin("com.eed3si9n"       % "sbt-assembly"       % "2.4.1")
addSbtPlugin("ch.epfl.scala"      % "sbt-scalafix"       % "0.14.7")
addSbtPlugin("org.scalameta"      % "sbt-scalafmt"       % "2.6.2")
addSbtPlugin("com.github.sbt"     % "sbt-java-formatter" % "0.13.1")
addSbtPlugin("pl.project13.scala" % "sbt-jmh"            % "0.4.8")
addSbtPlugin("org.scala-js"       % "sbt-scalajs"        % "1.22.0")
addSbtPlugin("org.scala-native"   % "sbt-scala-native"   % "0.5.12")

// XXX Scala.js's fullOptJS is dropping gcc, see https://www.scala-js.org/news/2026/04/04/announcing-scalajs-1.21.0
// v20240317 is the newest release targeting pre-JDK-21 bytecode; later releases require JDK 21.
// It currently produces a byte-for-byte identical pass bundle to v20260513, so the older compiler is safe here.
libraryDependencies += "com.google.javascript" % "closure-compiler" % "v20240317"
