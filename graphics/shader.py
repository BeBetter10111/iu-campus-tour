from pathlib import Path
import re
import numpy as np
from OpenGL.GL import (
    glCreateShader, glShaderSource, glCompileShader, glGetShaderiv, glGetShaderInfoLog,
    glCreateProgram, glAttachShader, glLinkProgram, glGetProgramiv, glGetProgramInfoLog,
    glDeleteShader, glUseProgram, glGetUniformLocation,
    glUniformMatrix4fv, glUniform3f, glUniform1f, glUniform1i,
    GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, GL_COMPILE_STATUS, GL_LINK_STATUS, GL_TRUE, glUniformMatrix3fv
)

SHADER_DIR = Path(__file__).resolve().parent.parent / "assets" / "shaders"


class Shader:
    def __init__(self, vert_src, frag_src):
        vs = self._compile(GL_VERTEX_SHADER, vert_src)
        fs = self._compile(GL_FRAGMENT_SHADER, frag_src)
        self.program = glCreateProgram()
        glAttachShader(self.program, vs)
        glAttachShader(self.program, fs)
        glLinkProgram(self.program)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise RuntimeError("Link error:\n" + glGetProgramInfoLog(self.program).decode())
        glDeleteShader(vs)   
        glDeleteShader(fs)
        self._uniforms = {}  

    @classmethod
    def from_files(cls, vert_name, frag_name):
        return cls((SHADER_DIR / vert_name).read_text(encoding="utf-8"),
                   (SHADER_DIR / frag_name).read_text(encoding="utf-8"))

    @staticmethod
    def _compile(kind, source):
        source = source.encode("ascii", "ignore").decode("ascii")

        sh = glCreateShader(kind)
        glShaderSource(sh, source)
        glCompileShader(sh)
        if not glGetShaderiv(sh, GL_COMPILE_STATUS):
            log = glGetShaderInfoLog(sh).decode(errors="ignore")
            lines = source.splitlines()
            context = ""
            for num in re.findall(r"\d+:(\d+)", log):
                n = int(num)
                if 1 <= n <= len(lines):
                    context += f"\n  dòng {n}: {lines[n - 1].strip()}"
            kind_name = "VERTEX" if kind == GL_VERTEX_SHADER else "FRAGMENT"
            raise RuntimeError(f"{kind_name} shader compile error:\n{log}{context}")
        return sh
    
    def use(self):
        glUseProgram(self.program)

    def _loc(self, name):
        if name not in self._uniforms:
            self._uniforms[name] = glGetUniformLocation(self.program, name)
        return self._uniforms[name]

    def set_mat4(self, name, m):
        glUniformMatrix4fv(self._loc(name), 1, GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))

    def set_vec3(self, name, x, y, z): glUniform3f(self._loc(name), x, y, z)
    def set_float(self, name, v): glUniform1f(self._loc(name), v)
    def set_int(self, name, v): glUniform1i(self._loc(name), v)
    def set_mat3(self, name, m):
        glUniformMatrix3fv(self._loc(name), 1, GL_TRUE, np.ascontiguousarray(m, dtype=np.float32))
