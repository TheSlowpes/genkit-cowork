# Changelog

## [2.0.0](https://github.com/TheSlowpes/genkit-cowork/compare/genkit-cowork-v1.0.0...genkit-cowork-v2.0.0) (2026-11-20)


### ⚠ BREAKING CHANGES

* **memory:** simplify memory around session assets
* **memory:** enforce tenant boundaries and async vector indexing

### Features

* **examples:** add pgvector file-chunk retrieval path ([cb97de0](https://github.com/TheSlowpes/genkit-cowork/commit/cb97de07e325b96b239a3c55004ae02c8ee168df))
* **examples:** wire memory retrieval tools into tui-chat ([71fd8de](https://github.com/TheSlowpes/genkit-cowork/commit/71fd8de1d92231baf97bdc80de0b0b962dd40cbc))
* **flows:** add BuildSystemPrompt / DefaultSystemPrompt ([31a29b3](https://github.com/TheSlowpes/genkit-cowork/commit/31a29b3275ce021deb07a3a9076af3930751cbdd))
* **flows:** add BuildSystemPrompt and DefaultSystemPrompt ([f2f7da2](https://github.com/TheSlowpes/genkit-cowork/commit/f2f7da2ca75549afe7e7772d48ef52917e10ac8d))
* **flows:** add BuildSystemPrompt and DefaultSystemPrompt ([9928d08](https://github.com/TheSlowpes/genkit-cowork/commit/9928d08f4d71c1db6f07c6f0232293a66dce2045))
* **flows:** drop tools list and skills section from system prompt ([d4a92fc](https://github.com/TheSlowpes/genkit-cowork/commit/d4a92fc138c621960a408159ed1e658598b57cc2))
* **flows:** drop tools list and skills section from system prompt ([17e3a6b](https://github.com/TheSlowpes/genkit-cowork/commit/17e3a6b7e2a8c21bad66aaa43d29b8366422ccc2))
* **flows:** use Genkit automatic agent loop ([c608e48](https://github.com/TheSlowpes/genkit-cowork/commit/c608e48f7b35cf1fad08416f0ae327f6b90db765))
* **genkit-cowork:** accept inline audio in Yzma chat ([186baa5](https://github.com/TheSlowpes/genkit-cowork/commit/186baa50012d98bd985f033ad6175631e5e5f7c0))
* **genkit-cowork:** accept inline video in Yzma chat ([bafc19f](https://github.com/TheSlowpes/genkit-cowork/commit/bafc19f3a6fc4b5d74a1813128cf29a1bd57ef37)), closes [#26](https://github.com/TheSlowpes/genkit-cowork/issues/26)
* **genkit-cowork:** discover local Yzma text models ([b25a68a](https://github.com/TheSlowpes/genkit-cowork/commit/b25a68a5843afd2a355fb9a1fcd0eb11925f7f98))
* **genkit-cowork:** embed text with local Yzma models ([27a5a37](https://github.com/TheSlowpes/genkit-cowork/commit/27a5a37eff351562bb26351a33e0a42c96fad8e3)), closes [#20](https://github.com/TheSlowpes/genkit-cowork/issues/20)
* **genkit-cowork:** return typed Yzma bounding boxes ([ef29bf6](https://github.com/TheSlowpes/genkit-cowork/commit/ef29bf6e76da1c843ab27e4106bcd8ae227bbb79)), closes [#24](https://github.com/TheSlowpes/genkit-cowork/issues/24)
* **genkit-cowork:** score typed local System One decisions ([8abdd80](https://github.com/TheSlowpes/genkit-cowork/commit/8abdd80dc7510c4c7d7c6df64aa9dfd99c1cda67)), closes [#23](https://github.com/TheSlowpes/genkit-cowork/issues/23)
* **genkit-cowork:** synthesize local speech with Yzma ([17a30b2](https://github.com/TheSlowpes/genkit-cowork/commit/17a30b2030381a7fce4de6aa4e9e99421f9e5677))
* **memory:** add append-only turn ledger and replay windows ([cf79f2a](https://github.com/TheSlowpes/genkit-cowork/commit/cf79f2af1638637e8ebebeb1fe8d2e1d36e4c52d))
* **memory:** add LLM-based consolidation and insight ledger ([d3f9f1b](https://github.com/TheSlowpes/genkit-cowork/commit/d3f9f1b69cbc24b68b4ef628baafaad252fe52d0))
* **memory:** add tenant preference records and consolidation promotion ([039dc60](https://github.com/TheSlowpes/genkit-cowork/commit/039dc60339f91c79d6e1f7e9bf2cfbf488d7f8e8))
* **memory:** add tenant-global file memory ingestion for structured text ([76ec0c6](https://github.com/TheSlowpes/genkit-cowork/commit/76ec0c65c891e570a7515be382a6e9114e25ccc8))
* **memory:** add tenant-scoped retrieval APIs and tool wrappers ([57af199](https://github.com/TheSlowpes/genkit-cowork/commit/57af199886b52834a3b3c6c67c24f85e693e3e9b))
* **memory:** align compliance for memory and pgvector sample ([2d6fe64](https://github.com/TheSlowpes/genkit-cowork/commit/2d6fe64818d9c67fca39de6044054c12dd1e5314))
* **memory:** align compliance for memory and pgvector sample ([5d3a3a8](https://github.com/TheSlowpes/genkit-cowork/commit/5d3a3a80ad9a46b4466ead788a41a627327ca771))
* **memory:** enforce tenant-scoped session operations ([08e7706](https://github.com/TheSlowpes/genkit-cowork/commit/08e77065dbff07e239793c6e307192b2640ed767))
* **memory:** use GenerationUsage for token-budget pruning ([04df281](https://github.com/TheSlowpes/genkit-cowork/commit/04df28199de7f7f859ded0c0ac5cf5be5a45d786))
* **skills:** rewrite plugin to single SkillTool with dynamic description, AllowedSkills, and default dir search ([0260450](https://github.com/TheSlowpes/genkit-cowork/commit/0260450f106b71be03ea5b4e9ffa1b409d47037c))
* **skills:** rewrite plugin to single SkillTool with dynamic description, AllowedSkills, and default dir search ([e435684](https://github.com/TheSlowpes/genkit-cowork/commit/e435684dee40433ddc3fe5300abc77ee843d3919))
* **tools,flows:** add tenant insight search tool and consolidation flow ([0f843bf](https://github.com/TheSlowpes/genkit-cowork/commit/0f843bf888b1765a690130e38295a4ed1553d125))
* **tools:** add find tool ([891f6ab](https://github.com/TheSlowpes/genkit-cowork/commit/891f6abb6c95f21170203540f2552559d4cd8ba4))
* **yzma:** add lazy model loading and lifecycle management ([0109ae6](https://github.com/TheSlowpes/genkit-cowork/commit/0109ae6bacd551e4e2fe432e3ce9a15eeee6f6ef))
* **yzma:** constrain structured generation ([1bbf5ab](https://github.com/TheSlowpes/genkit-cowork/commit/1bbf5ab4d7bdf312ecd22d895970acc657ecd468))
* **yzma:** separate model and generation configuration ([15f986f](https://github.com/TheSlowpes/genkit-cowork/commit/15f986f757d90f6f0f15c568234e476e34a74478))
* **yzma:** support inline images in local vision chat ([0d85932](https://github.com/TheSlowpes/genkit-cowork/commit/0d8593211bd1ba7d286aa07d462126b76d672380))
* **yzma:** support streaming chat and raw completion ([dc04b67](https://github.com/TheSlowpes/genkit-cowork/commit/dc04b673e0a5cfb0d589b0a0423aae04b89faed5))
* **yzma:** support tool calling and embedded Jinja templates ([ce8dfd7](https://github.com/TheSlowpes/genkit-cowork/commit/ce8dfd73dd90f2a4454e2e302e708f0f4a290ce9))


### Bug Fixes

* **examples:** stop TUI heartbeat on interrupt ([a290feb](https://github.com/TheSlowpes/genkit-cowork/commit/a290feba2f7951015f35f7faaf9dcbff7251587b))
* fixed documentation of genkit.Init signature and removed default max turns not being reached ([74b4766](https://github.com/TheSlowpes/genkit-cowork/commit/74b47662c05189e977eb76a07b7c1d57157992dd))
* **flows:** exclude system prompts from session history ([6b3b8d2](https://github.com/TheSlowpes/genkit-cowork/commit/6b3b8d2abbafac7e607c13ce5332099ca31fa562))
* **flows:** preserve model finish reason in agent loop ([9ef39a8](https://github.com/TheSlowpes/genkit-cowork/commit/9ef39a84c1b7d66550305ab296cdb360e6333dde))
* **flows:** require tenant and session IDs in flow inputs ([a9557cd](https://github.com/TheSlowpes/genkit-cowork/commit/a9557cd9e47a07376d1dbcae704819a674b0b4af))
* **genkit-cowork:** address Yzma final review findings ([44b0cba](https://github.com/TheSlowpes/genkit-cowork/commit/44b0cba249b1ee456eb6a1988819736b6692edaf))
* **genkit-cowork:** address Yzma parser review findings ([ae29688](https://github.com/TheSlowpes/genkit-cowork/commit/ae29688f2c8d47e6b85b753ac97170ece222d305))
* **genkit-cowork:** apply Pelican grounding image budget ([2ff99f9](https://github.com/TheSlowpes/genkit-cowork/commit/2ff99f9f7ddb19ee239413ba9ca6fdc3a57615f8))
* **genkit-cowork:** close Yzma parser boundary gaps ([4823101](https://github.com/TheSlowpes/genkit-cowork/commit/4823101b13e6333278842dffb5e9dbb3e45687f5))
* **genkit-cowork:** parse Harmony tool calls without coercion ([2996051](https://github.com/TheSlowpes/genkit-cowork/commit/29960519b092c90a9648a8636e02c1e13fd55a2e))
* **genkit-cowork:** preserve spaced GLM tool arguments ([1436962](https://github.com/TheSlowpes/genkit-cowork/commit/143696202ff34df9e784c297b90bbe5a01b27d53))
* **genkit-cowork:** preserve Yzma tool text and inputs ([69d8927](https://github.com/TheSlowpes/genkit-cowork/commit/69d89279d386b009b55950e862ae31b3e80ca400))
* **genkit-cowork:** reject folded Qwen call boundaries ([ce621ed](https://github.com/TheSlowpes/genkit-cowork/commit/ce621ed9f3e5638effbaf1c20396a2ac971d6edd))
* **genkit-cowork:** report both native grounding outcomes ([13108a7](https://github.com/TheSlowpes/genkit-cowork/commit/13108a7a64bc48554a1ef8e4a9cbf95cc69d652d))
* **genkit-cowork:** retain Yzma companion handles through rollback ([e4f250d](https://github.com/TheSlowpes/genkit-cowork/commit/e4f250debe16c823fc6306031fc42669f354974c))
* **genkit-cowork:** validate Yzma calls from shared frames ([090b924](https://github.com/TheSlowpes/genkit-cowork/commit/090b924a2a08912edb9d04b7eaf81059219b5b1b))
* **genkit-cowork:** validate Yzma tool calls before dispatch ([0404440](https://github.com/TheSlowpes/genkit-cowork/commit/04044402ef7649bda5f78eb21e3e246eef4e36a5))
* **memory:** add tenantID and fileID validation to PutFile ([82693dd](https://github.com/TheSlowpes/genkit-cowork/commit/82693dd420a2cc160f6ba9004e9712082748fd97))
* **memory:** enforce tenant boundaries and async vector indexing ([3fc143e](https://github.com/TheSlowpes/genkit-cowork/commit/3fc143e63fbe23d5b1ed2775a7fc0399504259f2))
* **memory:** remove rebase-duplicated session media helpers ([5f48de3](https://github.com/TheSlowpes/genkit-cowork/commit/5f48de34affa3431a313d117ee0c7ef8e7378768))
* **memory:** serialize vector backend operations in VectorOperator ([b20e435](https://github.com/TheSlowpes/genkit-cowork/commit/b20e43520192e2ce1613c252dc183df16a4fb529))
* **memory:** validate path segments in Put and PutFile to prevent path traversal ([625fad0](https://github.com/TheSlowpes/genkit-cowork/commit/625fad0c54773905c7e8a36f48b63238a5bf88de))
* **memory:** validate tenantID and fileID in PutFile ([c792617](https://github.com/TheSlowpes/genkit-cowork/commit/c79261754e850556af553c678a3716c824f46785))
* **memory:** validate tenantID and fileID in PutFile ([c17af02](https://github.com/TheSlowpes/genkit-cowork/commit/c17af02a2d9a9d57621c0a5652030b18fd82f440))
* **release:** remove invalid extra-files entry from release-please-config.json ([9f707ff](https://github.com/TheSlowpes/genkit-cowork/commit/9f707ffa5b15eff8099b59272693a4d5bffd0f29))
* **release:** remove invalid extra-files entry from release-please-config.json ([f2017fb](https://github.com/TheSlowpes/genkit-cowork/commit/f2017fb64183f3ec02c9e18f2c560e58a54242a4))
* **release:** remove invalid extra-files entry from release-please-config.json ([bc26846](https://github.com/TheSlowpes/genkit-cowork/commit/bc26846919da42821d661fb1cb4ee2335c7fac5a))
* **tools:** make TestResolveToCwd OS agnostic using filepath.Join and t.TempDir ([e6af328](https://github.com/TheSlowpes/genkit-cowork/commit/e6af328370ed02af64dfd1b85384edb1439b083b))
* **tools:** make TestResolveToCwd OS agnostic using filepath.Join and t.TempDir ([c4e0e29](https://github.com/TheSlowpes/genkit-cowork/commit/c4e0e29e181612cd5b4b1c5a5680a7f9a274ee7c))


### Documentation

* fix NewFileSessionOperator examples in README to pass both rootDir and tenantID ([d74a6a7](https://github.com/TheSlowpes/genkit-cowork/commit/d74a6a73a9b55a8e7f1f685b6ed61b27beec1416))
* **flows:** document Genkit automatic agent loop ([005e622](https://github.com/TheSlowpes/genkit-cowork/commit/005e62225d6a1cfa8131fb2b910e89b326a43b0c))
* **genkit-cowork:** standardize Yzma model download examples ([bdedc9f](https://github.com/TheSlowpes/genkit-cowork/commit/bdedc9fe857eff73e8b20ba670f9ad10f5c637fa))
* list plugins after framework documentation ([b1b7f9c](https://github.com/TheSlowpes/genkit-cowork/commit/b1b7f9ce9935b069405a6408f08874f1284bd907))
* **yzma:** record the native vision example response ([f0505e2](https://github.com/TheSlowpes/genkit-cowork/commit/f0505e2cd9d746f09b54e930b007ae481dbcc025))


### Tests

* **flows:** cover automatic loop persistence ([a58ef94](https://github.com/TheSlowpes/genkit-cowork/commit/a58ef942c7c988c7e3ccc26377d9d0bdbf847bec))
* **memory:** isolate vector load delegation from indexing ([9c5ba91](https://github.com/TheSlowpes/genkit-cowork/commit/9c5ba9106ee7989c1c77cf0c35278df843113829))
* **skills:** use go-cmp in TestParseSkillMetadata_FullMetadata ([fd13166](https://github.com/TheSlowpes/genkit-cowork/commit/fd13166c32aba940ff3e3a8819ea4a579641a629))
* **skills:** use go-cmp in TestParseSkillMetadata_FullMetadata ([719e0d3](https://github.com/TheSlowpes/genkit-cowork/commit/719e0d3bbdbbcf251d38315d69e331e3e8599a72))
* **skills:** use go-cmp in TestParseSkillMetadata_FullMetadata ([34f269f](https://github.com/TheSlowpes/genkit-cowork/commit/34f269fa2412a4f5d68c3689c224d9fd8e6b4684))
* **skills:** wire go-cmp into skills tests to justify direct dependency ([4e09ee5](https://github.com/TheSlowpes/genkit-cowork/commit/4e09ee5ae0239d367df042287c34bc68b71d9443))


### Refactoring

* **examples:** Made tui-chat use gemini embedding instead of hard-coded simple embedder ([64160cc](https://github.com/TheSlowpes/genkit-cowork/commit/64160cca4976bb1807207e7124490f31dc5849ed))
* **examples:** Separated pgvector example into more than one file for readability ([5980be7](https://github.com/TheSlowpes/genkit-cowork/commit/5980be732f668df252f6bbd246da8c968cf985b3))
* **memory:** rename LocalVecConfig TenantID to IndexDir ([1c2477d](https://github.com/TheSlowpes/genkit-cowork/commit/1c2477d7b200a5ebd1ffebc478aaccaa44f551ca))
* **memory:** simplify memory around session assets ([4049421](https://github.com/TheSlowpes/genkit-cowork/commit/4049421f029e394b62a5e67750d093b7e319a710))

## [1.0.0](https://github.com/TheSlowpes/genkit-cowork/compare/genkit-cowork-v0.1.0...genkit-cowork-v1.0.0) (2026-03-30)


### ⚠ BREAKING CHANGES

* **memory:** simplify memory around session assets
* **memory:** enforce tenant boundaries and async vector indexing

### Features

* **examples:** add pgvector file-chunk retrieval path ([cb97de0](https://github.com/TheSlowpes/genkit-cowork/commit/cb97de07e325b96b239a3c55004ae02c8ee168df))
* **examples:** wire memory retrieval tools into tui-chat ([71fd8de](https://github.com/TheSlowpes/genkit-cowork/commit/71fd8de1d92231baf97bdc80de0b0b962dd40cbc))
* **flows:** add BuildSystemPrompt / DefaultSystemPrompt ([31a29b3](https://github.com/TheSlowpes/genkit-cowork/commit/31a29b3275ce021deb07a3a9076af3930751cbdd))
* **flows:** add BuildSystemPrompt and DefaultSystemPrompt ([f2f7da2](https://github.com/TheSlowpes/genkit-cowork/commit/f2f7da2ca75549afe7e7772d48ef52917e10ac8d))
* **flows:** add BuildSystemPrompt and DefaultSystemPrompt ([9928d08](https://github.com/TheSlowpes/genkit-cowork/commit/9928d08f4d71c1db6f07c6f0232293a66dce2045))
* **flows:** drop tools list and skills section from system prompt ([d4a92fc](https://github.com/TheSlowpes/genkit-cowork/commit/d4a92fc138c621960a408159ed1e658598b57cc2))
* **flows:** drop tools list and skills section from system prompt ([17e3a6b](https://github.com/TheSlowpes/genkit-cowork/commit/17e3a6b7e2a8c21bad66aaa43d29b8366422ccc2))
* **memory:** add append-only turn ledger and replay windows ([cf79f2a](https://github.com/TheSlowpes/genkit-cowork/commit/cf79f2af1638637e8ebebeb1fe8d2e1d36e4c52d))
* **memory:** add LLM-based consolidation and insight ledger ([d3f9f1b](https://github.com/TheSlowpes/genkit-cowork/commit/d3f9f1b69cbc24b68b4ef628baafaad252fe52d0))
* **memory:** add tenant preference records and consolidation promotion ([039dc60](https://github.com/TheSlowpes/genkit-cowork/commit/039dc60339f91c79d6e1f7e9bf2cfbf488d7f8e8))
* **memory:** add tenant-global file memory ingestion for structured text ([76ec0c6](https://github.com/TheSlowpes/genkit-cowork/commit/76ec0c65c891e570a7515be382a6e9114e25ccc8))
* **memory:** add tenant-scoped retrieval APIs and tool wrappers ([57af199](https://github.com/TheSlowpes/genkit-cowork/commit/57af199886b52834a3b3c6c67c24f85e693e3e9b))
* **memory:** align compliance for memory and pgvector sample ([2d6fe64](https://github.com/TheSlowpes/genkit-cowork/commit/2d6fe64818d9c67fca39de6044054c12dd1e5314))
* **memory:** align compliance for memory and pgvector sample ([5d3a3a8](https://github.com/TheSlowpes/genkit-cowork/commit/5d3a3a80ad9a46b4466ead788a41a627327ca771))
* **memory:** enforce tenant-scoped session operations ([08e7706](https://github.com/TheSlowpes/genkit-cowork/commit/08e77065dbff07e239793c6e307192b2640ed767))
* **memory:** use GenerationUsage for token-budget pruning ([04df281](https://github.com/TheSlowpes/genkit-cowork/commit/04df28199de7f7f859ded0c0ac5cf5be5a45d786))
* **skills:** rewrite plugin to single SkillTool with dynamic description, AllowedSkills, and default dir search ([0260450](https://github.com/TheSlowpes/genkit-cowork/commit/0260450f106b71be03ea5b4e9ffa1b409d47037c))
* **skills:** rewrite plugin to single SkillTool with dynamic description, AllowedSkills, and default dir search ([e435684](https://github.com/TheSlowpes/genkit-cowork/commit/e435684dee40433ddc3fe5300abc77ee843d3919))
* **tools,flows:** add tenant insight search tool and consolidation flow ([0f843bf](https://github.com/TheSlowpes/genkit-cowork/commit/0f843bf888b1765a690130e38295a4ed1553d125))


### Bug Fixes

* **flows:** preserve model finish reason in agent loop ([9ef39a8](https://github.com/TheSlowpes/genkit-cowork/commit/9ef39a84c1b7d66550305ab296cdb360e6333dde))
* **flows:** require tenant and session IDs in flow inputs ([a9557cd](https://github.com/TheSlowpes/genkit-cowork/commit/a9557cd9e47a07376d1dbcae704819a674b0b4af))
* **memory:** add tenantID and fileID validation to PutFile ([82693dd](https://github.com/TheSlowpes/genkit-cowork/commit/82693dd420a2cc160f6ba9004e9712082748fd97))
* **memory:** enforce tenant boundaries and async vector indexing ([3fc143e](https://github.com/TheSlowpes/genkit-cowork/commit/3fc143e63fbe23d5b1ed2775a7fc0399504259f2))
* **memory:** remove rebase-duplicated session media helpers ([5f48de3](https://github.com/TheSlowpes/genkit-cowork/commit/5f48de34affa3431a313d117ee0c7ef8e7378768))
* **memory:** serialize vector backend operations in VectorOperator ([b20e435](https://github.com/TheSlowpes/genkit-cowork/commit/b20e43520192e2ce1613c252dc183df16a4fb529))
* **memory:** validate path segments in Put and PutFile to prevent path traversal ([625fad0](https://github.com/TheSlowpes/genkit-cowork/commit/625fad0c54773905c7e8a36f48b63238a5bf88de))
* **memory:** validate tenantID and fileID in PutFile ([c792617](https://github.com/TheSlowpes/genkit-cowork/commit/c79261754e850556af553c678a3716c824f46785))
* **memory:** validate tenantID and fileID in PutFile ([c17af02](https://github.com/TheSlowpes/genkit-cowork/commit/c17af02a2d9a9d57621c0a5652030b18fd82f440))
* **release:** remove invalid extra-files entry from release-please-config.json ([9f707ff](https://github.com/TheSlowpes/genkit-cowork/commit/9f707ffa5b15eff8099b59272693a4d5bffd0f29))
* **release:** remove invalid extra-files entry from release-please-config.json ([f2017fb](https://github.com/TheSlowpes/genkit-cowork/commit/f2017fb64183f3ec02c9e18f2c560e58a54242a4))
* **release:** remove invalid extra-files entry from release-please-config.json ([bc26846](https://github.com/TheSlowpes/genkit-cowork/commit/bc26846919da42821d661fb1cb4ee2335c7fac5a))
* **tools:** make TestResolveToCwd OS agnostic using filepath.Join and t.TempDir ([e6af328](https://github.com/TheSlowpes/genkit-cowork/commit/e6af328370ed02af64dfd1b85384edb1439b083b))
* **tools:** make TestResolveToCwd OS agnostic using filepath.Join and t.TempDir ([c4e0e29](https://github.com/TheSlowpes/genkit-cowork/commit/c4e0e29e181612cd5b4b1c5a5680a7f9a274ee7c))


### Documentation

* fix NewFileSessionOperator examples in README to pass both rootDir and tenantID ([d74a6a7](https://github.com/TheSlowpes/genkit-cowork/commit/d74a6a73a9b55a8e7f1f685b6ed61b27beec1416))


### Tests

* **skills:** use go-cmp in TestParseSkillMetadata_FullMetadata ([fd13166](https://github.com/TheSlowpes/genkit-cowork/commit/fd13166c32aba940ff3e3a8819ea4a579641a629))
* **skills:** use go-cmp in TestParseSkillMetadata_FullMetadata ([719e0d3](https://github.com/TheSlowpes/genkit-cowork/commit/719e0d3bbdbbcf251d38315d69e331e3e8599a72))
* **skills:** use go-cmp in TestParseSkillMetadata_FullMetadata ([34f269f](https://github.com/TheSlowpes/genkit-cowork/commit/34f269fa2412a4f5d68c3689c224d9fd8e6b4684))
* **skills:** wire go-cmp into skills tests to justify direct dependency ([4e09ee5](https://github.com/TheSlowpes/genkit-cowork/commit/4e09ee5ae0239d367df042287c34bc68b71d9443))


### Refactoring

* **examples:** Made tui-chat use gemini embedding instead of hard-coded simple embedder ([64160cc](https://github.com/TheSlowpes/genkit-cowork/commit/64160cca4976bb1807207e7124490f31dc5849ed))
* **examples:** Separated pgvector example into more than one file for readability ([5980be7](https://github.com/TheSlowpes/genkit-cowork/commit/5980be732f668df252f6bbd246da8c968cf985b3))
* **memory:** rename LocalVecConfig TenantID to IndexDir ([1c2477d](https://github.com/TheSlowpes/genkit-cowork/commit/1c2477d7b200a5ebd1ffebc478aaccaa44f551ca))
* **memory:** simplify memory around session assets ([4049421](https://github.com/TheSlowpes/genkit-cowork/commit/4049421f029e394b62a5e67750d093b7e319a710))
