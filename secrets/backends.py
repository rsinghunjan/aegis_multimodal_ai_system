132
133
134
135
136
137
138
139
140
141
142
143
144
145
146
147
148
149
150
151
152
153
154
155
156
157
158
159
160
161
162
163
164
165
166
167
168
169
170
171
172
173
174
175
176
177
178
179
180
181
182
"""
            self._aws_client = None
            return
        try:
            self._aws_client = boto3.client("secretsmanager")
        except Exception:
            logger.exception("Failed to create AWS Secrets Manager client")
            self._aws_client = None

    def _get_from_aws(self, name: str, key: Optional[str] = None) -> Optional[str]:
        self._init_aws()
        if not self._aws_client:
            return None
        try:
            resp = self._aws_client.get_secret_value(SecretId=name)
            secret_str = resp.get("SecretString")
            if secret_str:
                try:
                    parsed = json.loads(secret_str)
                    if key:
                        return parsed.get(key)
                    return json.dumps(parsed)
                except Exception:
                    # not JSON
                    if key:
                        return None
                    return secret_str
            # binary case
            return None
        except Exception:
            logger.debug("AWS Secrets Manager read failed for %s", name)
            return None

    def get_secret(self, name: str, key: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret named `name`. If secret is structured and you want a field, pass `key`.
        Tries configured backends in order; does not log secret contents.
        """
        for backend in self.prefer:
            if backend == "vault":
                val = self._get_from_vault(name, key=key)
                if val is not None:
                    return val
            elif backend == "aws":
                val = self._get_from_aws(name, key=key)
                if val is not None:
                    return val
            elif backend == "env":
                val = self._get_from_env(name)
                if val is not None:
                    return val
        return default
