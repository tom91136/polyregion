#include "catch.hpp"
#include "compiler.h"
#include "generated/polyast.h"
#include "generated/polyast_codec.h"
#include "polyregion_compiler.h"

using namespace polyregion::polyast;

TEST_CASE("json round-trip", "[ast]") {
  Function expected(                                        //
      "foo",                                                //
      {Named("a", Type::Int()), Named("b", Type::Float())}, //
      Type::Unit(),                                         //
      {
          Stmt::Comment("a"),                                     //
          Stmt::Comment("b"),                                     //
          Stmt::Break(),                                          //
          Stmt::Return(Expr::Alias(Term::IntConst(1))),           //
          Stmt::Cond(                                             //
              Expr::Alias(Term::BoolConst(true)),                 //
              {Stmt::Return(Expr::Alias(Term::IntConst(1)))},     //
              {Stmt::Return(Expr::Alias(Term::BoolConst(false)))} //
              )                                                   //
      },
      {} //
  );
  auto actual = function_from_json(function_to_json(expected));
  CHECK(expected == actual);
}

TEST_CASE("initialise more than once should work", "[compiler]") {
  polyregion_initialise();
  polyregion_initialise();
  polyregion_initialise();
  polyregion_initialise();
}

TEST_CASE("empty function should compile", "[compiler]") {
  polyregion_initialise();
  Function fn("foo", {}, Type::Unit(), {}, {});
  auto data = nlohmann::json::to_msgpack(function_to_json(fn));

  polyregion_buffer buffer{data.data(), data.size()};

  auto compilation = polyregion_compile(&buffer, true, POLYREGION_BACKEND_LLVM);

  polyregion_release_compile(compilation);
}

TEST_CASE("struct member access", "[compiler]") {
  polyregion_initialise();

  auto sdef = Sym({"a", "b"});
  StructDef def(sdef, {Named("x", Type::Int()), Named("y", Type::Int())});
  auto arg = Named("in", Type::Struct(sdef));
  Function fn("foo", {arg}, Type::Unit(),
              {
                  Stmt::Var(                    //
                      Named("y2", Type::Int()), //
                      {
                          Expr::Alias(Term::Select({arg}, Named("y", Type::Int()))) //
                      }                                                             //
                      ),
                  Stmt::Return(Expr::Alias(Term::UnitConst())),
              },
              {def});

  polyregion::compiler::compile(fn);
}