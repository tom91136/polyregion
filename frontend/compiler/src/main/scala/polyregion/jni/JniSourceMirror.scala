package polyregion.jni

import java.lang.reflect.{Constructor, Field, Method, Modifier}
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardOpenOption}
import scala.collection.immutable.Set
import scala.util.{Failure, Success, Try}

object JniSourceMirror {

  // JNI has specialisations for one reference type: String
  private def isNonJniSpecialisedObjectType(t: Class[_]): Boolean =
    !(t.getName == StringClass || t.isPrimitive || t.isArray)

  private def isVoid(c: Class[_]): Boolean = c == java.lang.Void.TYPE

  private def descriptor(m: Class[_]): String = m match {
    case x if x.equals(java.lang.Double.TYPE)    => "D"
    case x if x.equals(java.lang.Float.TYPE)     => "F"
    case x if x.equals(java.lang.Long.TYPE)      => "J"
    case x if x.equals(java.lang.Integer.TYPE)   => "I"
    case x if x.equals(java.lang.Short.TYPE)     => "S"
    case x if x.equals(java.lang.Character.TYPE) => "C"
    case x if x.equals(java.lang.Boolean.TYPE)   => "Z"
    case x if x.equals(java.lang.Byte.TYPE)      => "B"
    case x if x.equals(java.lang.Void.TYPE)      => "V"
    case x if x.isArray                          => x.getName.replace('.', '/')
    case x                                       => s"L${x.getName.replace('.', '/')};"
  }

  private def descriptor(c: Constructor[_]): String = s"(${c.getParameterTypes.map(descriptor(_)).mkString})"
  private def descriptor(m: Method): String =
    s"(${m.getParameterTypes.map(descriptor(_)).mkString})${descriptor(m.getReturnType)}"
  private def signature(m: Method): String = m.getName + descriptor(m)
  private def descriptorSafe(m: Method): String =
    m.getName + "_" + (m.getParameterTypes.map(descriptor(_)) :+ descriptor(m.getReturnType)).mkString
      .replace('/', '_')
      .replace(';', '_')
      .replace('[', 'a')
      .replace('$', '_')

  private def jniTypeName(t: Class[_]): String = t match {
    case x if x.equals(java.lang.Double.TYPE)                     => "jdouble"
    case x if x.equals(java.lang.Float.TYPE)                      => "jfloat"
    case x if x.equals(java.lang.Long.TYPE)                       => "jlong"
    case x if x.equals(java.lang.Integer.TYPE)                    => "jint"
    case x if x.equals(java.lang.Short.TYPE)                      => "jshort"
    case x if x.equals(java.lang.Character.TYPE)                  => "jcharacter"
    case x if x.equals(java.lang.Boolean.TYPE)                    => "jboolean"
    case x if x.equals(java.lang.Byte.TYPE)                       => "jbyte"
    case x if x.equals(java.lang.Void.TYPE)                       => "void"
    case x if x.isArray && x.componentType.getName == StringClass => s"jobjectArray" // jstringArray is not a thing
    case x if x.isArray                                           => s"${jniTypeName(x.getComponentType)}Array"
    case x if x.getName == StringClass                            => "jstring"
    case _                                                        => "jobject"
  }

  private def jniTypedFunctionName(t: Class[_]): String = t match {
    case x if x.equals(java.lang.Double.TYPE)    => "Double"
    case x if x.equals(java.lang.Float.TYPE)     => "Float"
    case x if x.equals(java.lang.Long.TYPE)      => "Long"
    case x if x.equals(java.lang.Integer.TYPE)   => "Int"
    case x if x.equals(java.lang.Short.TYPE)     => "Short"
    case x if x.equals(java.lang.Character.TYPE) => "Char"
    case x if x.equals(java.lang.Boolean.TYPE)   => "Boolean"
    case x if x.equals(java.lang.Byte.TYPE)      => "Byte"
    case x if x.equals(java.lang.Void.TYPE)      => "Void"
    case _                                       => "Object" // even string needs a upcast
  }

  private final val StringClass                  = classOf[String].getName
  private final val ObjectClassMethodsSignatures = classOf[AnyRef].getDeclaredMethods.map(m => signature(m)).toSet

  private def reflectJniSource(
      knownClasses: Set[String],
      cls: Class[_],
      fp: Field => Boolean,
      cp: Constructor[_] => Boolean,
      mp: Method => Boolean
  ): (String, String) = {

    val publicFields =
      cls.getDeclaredFields
        .filterNot { m =>
          val mod = m.getModifiers
          Modifier.isPrivate(mod) || Modifier.isProtected(mod) || Modifier.isNative(mod) || m.isSynthetic
        }
        .filter(fp)
        .map(f => f.getName -> f.getType)
        .toList

    val publicMethods = cls.getDeclaredMethods
      .filterNot { m =>
        val mod = m.getModifiers
        Modifier.isPrivate(mod) || Modifier.isProtected(mod) || Modifier.isNative(mod) || m.isSynthetic
      }
      .filterNot(m => ObjectClassMethodsSignatures.contains(signature(m)))
      .filter(mp)
      .toList

    val ctors = cls.getDeclaredConstructors
      .filter { m =>
        //        val mod = m.getModifiers
        !m.isSynthetic
      }
      .filter(cp)
      .toList

    publicFields.map { case (name, tpe) =>
      s"""static jfieldID $name = env->GetFieldID(clazz, "$name", "${descriptor(tpe)}");"""
    }

    val jniName = cls.getName.replace('.', '/')
    val clsName = cls.getSimpleName

    val metaStructMembers =
      "jclass" -> "clazz"
        :: publicFields.map { case (name, _) => s"jfieldID" -> s"${name}Field" }
        ::: ctors.zipWithIndex.map { case (_, i) => s"jmethodID" -> s"ctor${i}Method" }
        ::: publicMethods.map(f => s"jmethodID" -> s"${descriptorSafe(f)}Method")

    val metaStructMemberInits =
      ("clazz", s"""reinterpret_cast<jclass>(env->NewGlobalRef(env->FindClass("$jniName")))""", cls.getName)
        :: publicFields.map { case (name, tpe) =>
          (s"${name}Field", s"""env->GetFieldID(clazz, "$name", "${descriptor(tpe)}")""", tpe.getName)
        }
        ::: ctors.zipWithIndex.map { case (ctor, i) =>
          // ctor is (...)V, note the void at the end
          (s"ctor${i}Method", s"""env->GetMethodID(clazz, "<init>", "${descriptor(ctor)}V")""", ctor.toString)
        }
        ::: publicMethods.map { m =>
          val mod = if (Modifier.isStatic(m.getModifiers)) "Static" else ""
          (
            s"${descriptorSafe(m)}Method",
            s"""env->Get${mod}MethodID(clazz, "${m.getName}", "${descriptor(m)}")""",
            m.toString
          )
        }

    val (metaStructStaticMethodFns, metaStructStaticMethodFnImpls) =
      publicMethods
        .filter(m => Modifier.isStatic(m.getModifiers))
        .map { m =>
          val name = m.getName
          val args =
            "JNIEnv *env" :: m.getParameters
              .map(p => s"${jniTypeName(p.getType)} ${p.getName}")
              .toList

          val implArgAps =
            ("clazz" :: s"${descriptorSafe(m)}Method" :: m.getParameters.map(_.getName).toList).mkString(", ")

          val proto  = s"${jniTypeName(m.getReturnType)} $name(${args.mkString(", ")}) const;"
          val rtnMod = jniTypedFunctionName(m.getReturnType)
          val rtn    = if (isVoid(m.getReturnType)) "" else "return "
          val impl =
            s"${jniTypeName(m.getReturnType)} $clsName::$name(${args.mkString(", ")}) const { ${rtn}env->CallStatic${rtnMod}Method($implArgAps); }"
          proto -> impl
        }
        .unzip

    val (instStructCtorFactoryFns, instStructCtorFactoryFnImpls) = ctors.zipWithIndex.map { case (ctor, i) =>
      val args = "JNIEnv *env" :: ctor.getParameters.toList.map { p =>
        s"${jniTypeName(p.getType)} ${p.getName}"
      }
      val implArgAps = s"ctor${i}Method" :: ctor.getParameters.map(p => s"${p.getName}").toList

      val proto = s"Instance operator()(${args.mkString(", ")}) const;"
      val impl = s"""$clsName::Instance $clsName::operator()(${args.mkString(", ")}) const {
                    |  return {*this, env->NewObject(clazz, ${implArgAps.mkString(", ")})};
                    |}""".stripMargin

      proto -> impl
    }.unzip

    def delegatedName(tpe: Class[_]) =
      if (!isNonJniSpecialisedObjectType(tpe)) None
      else {
        val ref = tpe
        if (knownClasses.contains(ref.getName)) Some(tpe.getSimpleName)
        else {
          Console.err.println(ref)
          None
        }
      }

    val (instStructFieldGetterFns, instStructFieldGetterFnImpls) =
      publicFields.map { case (field, tpe) =>
        val tpeName = jniTypeName(tpe)
        val getter  = s"env->Get${jniTypedFunctionName(tpe)}Field(instance, meta.${field}Field)"

        delegatedName(tpe) match {
          case Some(cls) =>
            s"$cls::Instance $field(JNIEnv *env, const $cls& clazz) const;" ->
              s"$cls::Instance $clsName::Instance::$field(JNIEnv *env, const $cls& clazz_) const { return {clazz_, $getter}; }"
          case None =>
            val value = tpe match {
              case s if s.getName == StringClass => s"reinterpret_cast<jstring>($getter)"
              case a if a.isArray                => s"reinterpret_cast<$tpeName>($getter)"
              case _                             => getter
            }

            s"$tpeName $field(JNIEnv *env) const;" ->
              s"$tpeName $clsName::Instance::$field(JNIEnv *env) const { return $value; }"
        }
      }.unzip

    val (instStructMethodFns, instStructMethodFnImpls) =
      publicMethods
        .filterNot(m => Modifier.isStatic(m.getModifiers))
        .map { m =>
          val name = m.getName
          val tpe  = m.getReturnType
          val args =
            "JNIEnv *env" :: m.getParameters
              .map(p => s"${jniTypeName(p.getType)} ${p.getName}")
              .toList
          val rtnMod = jniTypedFunctionName(tpe)

          val getter = s"env->Call${rtnMod}Method(instance, meta.${descriptorSafe(m)}Method)"

          delegatedName(tpe) match {
            case Some(cls) =>
              val argAndCls = (args :+ s"const $cls& clazz_").mkString(", ")
              s"$cls::Instance $name($argAndCls) const;" ->
                s"$cls::Instance $clsName::Instance::$name($argAndCls) const { return {clazz_, $getter}; }"
            case None =>
              val tpeName = jniTypeName(tpe)

              val value = tpe match {
                case s if s.getName == StringClass => s"reinterpret_cast<jstring>($getter)"
                case a if a.isArray                => s"reinterpret_cast<$tpeName>($getter)"
                case _                             => getter
              }

              val rtn     = if (isVoid(tpe)) "" else "return "
              val justArg = args.mkString(", ")
              s"$tpeName $name($justArg) const;" ->
                s"$tpeName $clsName::Instance::$name($justArg) const { $rtn$value; }"
          }
        }
        .unzip

    val structPrototype = //
      s"""struct $clsName {
         |  struct Instance {
         |    const $clsName &meta;
         |    jobject instance;
         |    Instance(const $clsName &meta, jobject instance);
		 |    template <typename T, typename F> std::optional<T> map(F && f) { return instance ? std::make_optional(f(*this)) : std::nullopt; };
         |    ${(instStructFieldGetterFns ++ instStructMethodFns).mkString("\n    ")}
         |  };
         |${metaStructMembers.map((t, n) => s"  $t $n;").mkString("\n")}
		 |private:
         |  explicit $clsName(JNIEnv *env);
		 |  static thread_local std::unique_ptr<$clsName> cached;
		 |public:
         |  static $clsName& of(JNIEnv *env);
         |  static void drop(JNIEnv *env);
         |  Instance wrap (JNIEnv *env, jobject instance);
         |  ${(metaStructStaticMethodFns ++ instStructCtorFactoryFns).mkString("\n  ")}
         |};""".stripMargin

    val structImpl =
      s"""$clsName::Instance::Instance(const $clsName &meta, jobject instance) : meta(meta), instance(instance) {}
         |${(instStructFieldGetterFnImpls ++ instStructMethodFnImpls).mkString("\n")}
         |$clsName::$clsName(JNIEnv *env)
         |    : ${metaStructMemberInits
        .map { case (m, v, comment) => s"$m($v)" }
        .mkString(s",\n      ")} { };
		 |thread_local std::unique_ptr<$clsName> $clsName::cached = {};
		 |$clsName& $clsName::of(JNIEnv *env) {
         |  if(!cached) cached = std::unique_ptr<$clsName>(new $clsName(env));
		 |  return *cached;
         |}
		 |void $clsName::drop(JNIEnv *env){
		 |  if(cached) {
		 |    env->DeleteGlobalRef(cached->clazz);
		 |    cached.reset();
		 |  }
		 |}
		 |$clsName::Instance $clsName::wrap(JNIEnv *env, jobject instance) { return {*this, instance}; }
         |${(metaStructStaticMethodFnImpls ++ instStructCtorFactoryFnImpls).mkString("\n")}
         |""".stripMargin

    println(structPrototype + "\n" + structImpl)

    structPrototype -> structImpl
  }

  @main def main(): Unit = {

    println("Generating C++ mirror for JNI...")

    val pending: List[(Class[_], Field => Boolean, Constructor[_] => Boolean, Method => Boolean)] =
      List(
        (
          classOf[java.nio.ByteBuffer],
          _ => false,
          _ => false,
          m => Set("allocate", "allocateDirect").contains(m.getName)
        ),
        (classOf[polyregion.jvm.runtime.Property], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Dim3], _ => true, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Policy], _ => true, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Device.Queue], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Device], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.runtime.Runtime], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Event], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Layout], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Member], _ => false, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Options], _ => true, _ => true, _ => false),
        (classOf[polyregion.jvm.compiler.Compilation], _ => false, _ => true, _ => false),
        (classOf[java.lang.String], _ => false, _ => false, _ => false),
        (classOf[java.lang.Runnable], _ => true, _ => true, _ => true)
      )

    val knownClasses: Set[String] = pending.map(_._1.getName).toSet

    val (headers, impls) = pending.map(reflectJniSource(knownClasses, _, _, _, _)).unzip

    val header =
      s"""#include <jni.h>
		 |#include <optional>
		 |#include <memory>
		 |namespace polyregion::generated {
	     |${headers.mkString("\n")}
         |}// polyregion::generated""".stripMargin

    val impl =
      s"""#include "mirror.h"
		 |using namespace polyregion::generated;
         |${impls.mkString("\n")}
         |""".stripMargin

    println(s"Generated ADT=${(header + impl).count(_ == '\n')} lines")

    val target = Paths.get("../native/bindings/jvm/generated/").toAbsolutePath.normalize

    println(s"Writing to $target")

    Files.createDirectories(target)

    def overwrite(path: Path)(content: String) = Files.write(
      path,
      content.getBytes(StandardCharsets.UTF_8),
      StandardOpenOption.TRUNCATE_EXISTING,
      StandardOpenOption.CREATE,
      StandardOpenOption.WRITE
    )

    overwrite(target.resolve("mirror.h"))(header)
    overwrite(target.resolve("mirror.cpp"))(impl)

    println("Done")
    ()
  }

}
