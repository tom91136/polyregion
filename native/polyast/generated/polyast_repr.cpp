#include "polyast_repr.h"
#include "aspartame/all.hpp"
#include "fmt/core.h"

using namespace aspartame;
using namespace std::string_literals;

namespace polyregion::polyast {

std::string repr(const SourcePosition& t) {  
  return fmt::format("{}:{}{}", t.file, t.line, t.col ^ map([&](const int32_t& c){ return fmt::format(":{}", c); }) ^ get_or_else(""s));
} 


std::string repr(const TypeSpace::Any& t) {  
  return [&]{
    if (t.is<TypeSpace::Global>()) { return "Global"s; }
    if (t.is<TypeSpace::Local>()) { return "Local"s; }
    if (t.is<TypeSpace::Private>()) { return "Private"s; }
  
    throw std::logic_error(fmt::format("Unhandled match case for t (of type TypeSpace::Any) at {}:{})", __FILE__, __LINE__));
  }();
} 


std::string repr(const TypeKind::Any& k) {  
  return [&]{
    if (k.is<TypeKind::None>()) { return "None"s; }
    if (k.is<TypeKind::Ref>()) { return "Ref"s; }
    if (k.is<TypeKind::Integral>()) { return "Integral"s; }
    if (k.is<TypeKind::Fractional>()) { return "Fractional"s; }
  
    throw std::logic_error(fmt::format("Unhandled match case for k (of type TypeKind::Any) at {}:{})", __FILE__, __LINE__));
  }();
} 


std::string repr(const Type::Any& t) {  
  return [&]{
    if (t.is<Type::Float16>()) { return "F16"s; }
    if (t.is<Type::Float32>()) { return "F32"s; }
    if (t.is<Type::Float64>()) { return "F64"s; }
    if (t.is<Type::IntU8>()) { return "U8"s; }
    if (t.is<Type::IntU16>()) { return "U16"s; }
    if (t.is<Type::IntU32>()) { return "U32"s; }
    if (t.is<Type::IntU64>()) { return "U64"s; }
    if (t.is<Type::IntS8>()) { return "I8"s; }
    if (t.is<Type::IntS16>()) { return "I16"s; }
    if (t.is<Type::IntS32>()) { return "I32"s; }
    if (t.is<Type::IntS64>()) { return "I64"s; }
    if (t.is<Type::Nothing>()) { return "Nothing"s; }
    if (t.is<Type::Unit0>()) { return "Unit0"s; }
    if (t.is<Type::Bool1>()) { return "Bool1"s; }
    if (auto _x = t.get<Type::Struct>()) {
      return fmt::format("{}", _x->name);
    }
    if (auto _x = t.get<Type::Ptr>()) {
      return fmt::format("{}[{}]^{}", repr(_x->comp), _x->length ^ map([&](const int32_t& __1){ return std::to_string(__1); }) ^ get_or_else(""s), repr(_x->space));
    }
    if (auto _x = t.get<Type::Annotated>()) {
      return fmt::format("{}{}{}", repr(_x->tpe), _x->pos ^ map([&](const SourcePosition& s){ return fmt::format("/*{}*/", repr(s)); }) ^ get_or_else(""s), _x->comment ^ map([&](const std::string& s){ return fmt::format("/*{}*/", s); }) ^ get_or_else(""s));
    }
  
    throw std::logic_error(fmt::format("Unhandled match case for t (of type Type::Any) at {}:{})", __FILE__, __LINE__));
  }();
} 


std::string repr(const Named& n) {  
  return fmt::format("{}", n.symbol);
} 


std::string repr(const Expr::Any& e) {  
  return [&]{
    if (auto _x = e.get<Expr::Float16Const>()) {
      return fmt::format("f16({})", _x->value);
    }
    if (auto _x = e.get<Expr::Float32Const>()) {
      return fmt::format("f32({})", _x->value);
    }
    if (auto _x = e.get<Expr::Float64Const>()) {
      return fmt::format("f64({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntU8Const>()) {
      return fmt::format("u8({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntU16Const>()) {
      return fmt::format("u16({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntU32Const>()) {
      return fmt::format("u32({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntU64Const>()) {
      return fmt::format("u64({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntS8Const>()) {
      return fmt::format("i8({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntS16Const>()) {
      return fmt::format("i16({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntS32Const>()) {
      return fmt::format("i32({})", _x->value);
    }
    if (auto _x = e.get<Expr::IntS64Const>()) {
      return fmt::format("i64({})", _x->value);
    }
    if (e.is<Expr::Unit0Const>()) { return "unit0(())"s; }
    if (auto _x = e.get<Expr::Bool1Const>()) {
      return fmt::format("bool1({})", _x->value);
    }
    if (auto _x = e.get<Expr::NullPtrConst>()) {
      return fmt::format("nullptr[{}, {}]", repr(_x->comp), repr(_x->space));
    }
    if (auto _x = e.get<Expr::SpecOp>()) {
      return [&]{
      if (_x->op.is<Spec::Assert>()) { return "'assert"s; }
      if (_x->op.is<Spec::GpuBarrierGlobal>()) { return "'gpuBarrierGlobal"s; }
      if (_x->op.is<Spec::GpuBarrierLocal>()) { return "'gpuBarrierLocal"s; }
      if (_x->op.is<Spec::GpuBarrierAll>()) { return "'gpuBarrierAll"s; }
      if (_x->op.is<Spec::GpuFenceGlobal>()) { return "'gpuFenceGlobal"s; }
      if (_x->op.is<Spec::GpuFenceLocal>()) { return "'gpuFenceLocal"s; }
      if (_x->op.is<Spec::GpuFenceAll>()) { return "'gpuFenceAll"s; }
      if (auto _z = _x->op.get<Spec::GpuGlobalIdx>()) {
        return fmt::format("'gpuGlobalIdx({})", repr(_z->dim));
      }
      if (auto _z = _x->op.get<Spec::GpuGlobalSize>()) {
        return fmt::format("'gpuGlobalSize({})", repr(_z->dim));
      }
      if (auto _z = _x->op.get<Spec::GpuGroupIdx>()) {
        return fmt::format("'gpuGroupIdx({})", repr(_z->dim));
      }
      if (auto _z = _x->op.get<Spec::GpuGroupSize>()) {
        return fmt::format("'gpuGroupSize({})", repr(_z->dim));
      }
      if (auto _z = _x->op.get<Spec::GpuLocalIdx>()) {
        return fmt::format("'gpuLocalIdx({})", repr(_z->dim));
      }
      if (auto _z = _x->op.get<Spec::GpuLocalSize>()) {
        return fmt::format("'gpuLocalSize({})", repr(_z->dim));
      }
    
      throw std::logic_error(fmt::format("Unhandled match case for _x->op (of type Spec::Any) at {}:{})", __FILE__, __LINE__));
    }();
    }
    if (auto _x = e.get<Expr::MathOp>()) {
      return [&]{
      if (auto _z = _x->op.get<Math::Abs>()) {
        return fmt::format("'abs({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Sin>()) {
        return fmt::format("'sin({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Cos>()) {
        return fmt::format("'cos({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Tan>()) {
        return fmt::format("'tan({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Asin>()) {
        return fmt::format("'asin({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Acos>()) {
        return fmt::format("'acos({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Atan>()) {
        return fmt::format("'atan({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Sinh>()) {
        return fmt::format("'sinh({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Cosh>()) {
        return fmt::format("'cosh({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Tanh>()) {
        return fmt::format("'tanh({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Signum>()) {
        return fmt::format("'signum({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Round>()) {
        return fmt::format("'round({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Ceil>()) {
        return fmt::format("'ceil({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Floor>()) {
        return fmt::format("'floor({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Rint>()) {
        return fmt::format("'rint({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Sqrt>()) {
        return fmt::format("'sqrt({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Cbrt>()) {
        return fmt::format("'cbrt({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Exp>()) {
        return fmt::format("'exp({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Expm1>()) {
        return fmt::format("'expm1({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Log>()) {
        return fmt::format("'log({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Log1p>()) {
        return fmt::format("'log1p({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Log10>()) {
        return fmt::format("'log10({})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Math::Pow>()) {
        return fmt::format("'pow({}, {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Math::Atan2>()) {
        return fmt::format("'atan2({}, {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Math::Hypot>()) {
        return fmt::format("'hypot({}, {})", repr(_z->x), repr(_z->y));
      }
    
      throw std::logic_error(fmt::format("Unhandled match case for _x->op (of type Math::Any) at {}:{})", __FILE__, __LINE__));
    }();
    }
    if (auto _x = e.get<Expr::IntrOp>()) {
      return [&]{
      if (auto _z = _x->op.get<Intr::BNot>()) {
        return fmt::format("('~{})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Intr::LogicNot>()) {
        return fmt::format("('!{})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Intr::Pos>()) {
        return fmt::format("('+{})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Intr::Neg>()) {
        return fmt::format("('-{})", repr(_z->x));
      }
      if (auto _z = _x->op.get<Intr::Add>()) {
        return fmt::format("({} '+ {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::Sub>()) {
        return fmt::format("({} '- {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::Mul>()) {
        return fmt::format("({} '* {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::Div>()) {
        return fmt::format("({} '/ {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::Rem>()) {
        return fmt::format("({} '% {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::Min>()) {
        return fmt::format("'min({}, {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::Max>()) {
        return fmt::format("'max({}, {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::BAnd>()) {
        return fmt::format("({} '& {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::BOr>()) {
        return fmt::format("({} '| {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::BXor>()) {
        return fmt::format("({} '^ {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::BSL>()) {
        return fmt::format("({} '<< {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::BSR>()) {
        return fmt::format("({} '>> {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::BZSR>()) {
        return fmt::format("({} '>>> {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicAnd>()) {
        return fmt::format("({} '&& {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicOr>()) {
        return fmt::format("({} '|| {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicEq>()) {
        return fmt::format("({} '== {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicNeq>()) {
        return fmt::format("({} '!= {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicLte>()) {
        return fmt::format("({} '<= {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicGte>()) {
        return fmt::format("({} '>= {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicLt>()) {
        return fmt::format("({} '< {})", repr(_z->x), repr(_z->y));
      }
      if (auto _z = _x->op.get<Intr::LogicGt>()) {
        return fmt::format("({} '> {})", repr(_z->x), repr(_z->y));
      }
    
      throw std::logic_error(fmt::format("Unhandled match case for _x->op (of type Intr::Any) at {}:{})", __FILE__, __LINE__));
    }();
    }
    if (auto _x = e.get<Expr::Select>()) {
      return (_x->init | append(_x->last) | map([&](const Named& x){ return fmt::format("{}: {}", x.symbol, repr(x.tpe)); }) | reduce([&](const std::string& acc, const std::string& x){ return fmt::format("({}).{}", acc, x); })) ^ get_or_else(""s);
    }
    if (auto _x = e.get<Expr::Poison>()) {
      return fmt::format("??? /*poison of type {}*/", repr(_x->t));
    }
    if (auto _x = e.get<Expr::Cast>()) {
      return fmt::format("({}).to[{}]", repr(_x->from), repr(_x->as));
    }
    if (auto _x = e.get<Expr::Index>()) {
      return fmt::format("({}).index[{}]({})", repr(_x->lhs), repr(_x->comp), repr(_x->idx));
    }
    if (auto _x = e.get<Expr::RefTo>()) {
      return fmt::format("({}).refTo[{}, {}]({})", repr(_x->lhs), repr(_x->comp), repr(_x->space), _x->idx ^ map([&](const Expr::Any& __2){ return repr(__2); }) ^ get_or_else(""s));
    }
    if (auto _x = e.get<Expr::Alloc>()) {
      return fmt::format("alloc[{}, {}]({})", repr(_x->comp), repr(_x->space), repr(_x->size));
    }
    if (auto _x = e.get<Expr::Invoke>()) {
      return fmt::format("{}({}): {}", _x->name, (_x->args | map([&](const Expr::Any& __3){ return repr(__3); }) | mk_string(", "s)), repr(_x->rtn));
    }
    if (auto _x = e.get<Expr::Annotated>()) {
      return fmt::format("{}{}{}", repr(_x->expr), _x->pos ^ map([&](const SourcePosition& s){ return fmt::format("/*{}*/", repr(s)); }) ^ get_or_else(""s), _x->comment ^ map([&](const std::string& s){ return fmt::format("/*{}*/", s); }) ^ get_or_else(""s));
    }
  
    throw std::logic_error(fmt::format("Unhandled match case for e (of type Expr::Any) at {}:{})", __FILE__, __LINE__));
  }();
} 


std::string repr(const Stmt::Any& stmt) {  
  return [&]{
    if (auto _x = stmt.get<Stmt::Block>()) {
      return fmt::format("{}\n{}\n{}", "{"s, (_x->stmts | map([&](const Stmt::Any& __4){ return repr(__4); }) | mk_string("\n"s)) ^ indent(2), "}"s);
    }
    if (auto _x = stmt.get<Stmt::Comment>()) {
      return fmt::format(" /* {} */", _x->value);
    }
    if (auto _x = stmt.get<Stmt::Var>()) {
      return fmt::format("var {}: {} = {}", _x->name.symbol, repr(_x->name.tpe), _x->expr ^ map([&](const Expr::Any& __5){ return repr(__5); }) ^ get_or_else("_"s));
    }
    if (auto _x = stmt.get<Stmt::Mut>()) {
      return fmt::format("{} = {}", repr(_x->name), repr(_x->expr));
    }
    if (auto _x = stmt.get<Stmt::Update>()) {
      return fmt::format("({}).update({}) = {}", repr(_x->lhs), repr(_x->idx), repr(_x->value));
    }
    if (auto _x = stmt.get<Stmt::While>()) {
      return fmt::format("while({}{}{}){}\n{}\n{}", "{"s, (_x->tests | map([&](const Stmt::Any& __6){ return repr(__6); }) | append(repr(_x->cond)) | mk_string(";"s)), "}"s, "{"s, (_x->body | map([&](const Stmt::Any& __7){ return repr(__7); }) | mk_string("\n"s)) ^ indent(2), "}"s);
    }
    if (auto _x = stmt.get<Stmt::ForRange>()) {
      return fmt::format("for({} = {}; {} < {}; {} += {}){}\n{}\n{}", repr(_x->induction), repr(_x->lbIncl), repr(_x->induction), repr(_x->ubExcl), repr(_x->induction), repr(_x->step), "{"s, (_x->body | map([&](const Stmt::Any& __8){ return repr(__8); }) | mk_string("\n"s)) ^ indent(2), "}"s);
    }
    if (stmt.is<Stmt::Break>()) { return "break;"s; }
    if (stmt.is<Stmt::Cont>()) { return "continue;"s; }
    if (auto _x = stmt.get<Stmt::Return>()) {
      return fmt::format("return {}", repr(_x->value));
    }
    if (auto _x = stmt.get<Stmt::Cond>()) {
      return fmt::format("if({}) {}\n{}{} else {}\n{}{}", repr(_x->cond), "{"s, (_x->trueBr | map([&](const Stmt::Any& __9){ return repr(__9); }) | mk_string("\n"s)) ^ indent(2), "}"s, "{"s, (_x->falseBr | map([&](const Stmt::Any& __10){ return repr(__10); }) | mk_string("\n"s)) ^ indent(2), "}"s);
    }
    if (auto _x = stmt.get<Stmt::Annotated>()) {
      return fmt::format("{}{}{}", repr(_x->stmt), _x->pos ^ map([&](const SourcePosition& s){ return fmt::format("/*{}*/", repr(s)); }) ^ get_or_else(""s), _x->comment ^ map([&](const std::string& s){ return fmt::format("/*{}*/", s); }) ^ get_or_else(""s));
    }
  
    throw std::logic_error(fmt::format("Unhandled match case for stmt (of type Stmt::Any) at {}:{})", __FILE__, __LINE__));
  }();
} 


std::string repr(const Arg& a) {  
  return fmt::format("{}: {}{}", a.named.symbol, repr(a.named.tpe), a.pos ^ map([&](const SourcePosition& s){ return fmt::format("/*{}*/", repr(s)); }) ^ get_or_else(""s));
} 


std::string repr(const FunctionAttr::Any& a) {  
  return [&]{
    if (a.is<FunctionAttr::Internal>()) { return "Internal"s; }
    if (a.is<FunctionAttr::Exported>()) { return "Exported"s; }
    if (a.is<FunctionAttr::FPRelaxed>()) { return "FPRelaxed"s; }
    if (a.is<FunctionAttr::FPStrict>()) { return "FPStrict"s; }
    if (a.is<FunctionAttr::Entry>()) { return "Entry"s; }
  
    throw std::logic_error(fmt::format("Unhandled match case for a (of type FunctionAttr::Any) at {}:{})", __FILE__, __LINE__));
  }();
} 


std::string repr(const Signature& f) {  
  return fmt::format("def {}({}: {}", f.name, (f.args | map([&](const Type::Any& __11){ return repr(__11); }) | mk_string(", "s)), repr(f.rtn));
} 


std::string repr(const Function& f) {  
  return fmt::format("def {}({}): {} /*{}*/ {}\n{}\n{}", f.name, (f.args | map([&](const Arg& a){ return fmt::format("{}: {}{}", a.named.symbol, repr(a.named.tpe), a.pos ^ map([&](const SourcePosition& s){ return fmt::format("/*{}*/", repr(s)); }) ^ get_or_else(""s)); }) | mk_string(", "s)), repr(f.rtn), (f.attrs | map([&](const FunctionAttr::Any& __12){ return repr(__12); }) | mk_string(", "s)), "{"s, (f.body | map([&](const Stmt::Any& __13){ return repr(__13); }) | mk_string("\n"s)) ^ indent(2), "}"s);
} 


std::string repr(const StructDef& s) {  
  return fmt::format("class {}({})", s.name, (s.members | map([&](const Named& m){ return fmt::format("{}: {}", m.symbol, repr(m.tpe)); }) | mk_string(", "s)));
} 


std::string repr(const Program& s) {  
  return fmt::format("{}\n{}", (s.structs | map([&](const StructDef& __14){ return repr(__14); }) | mk_string("\n"s)), (s.functions | map([&](const Function& __15){ return repr(__15); }) | mk_string("\n"s)));
} 

}
