CREATE TABLE public."metrics"
(
   y_name character varying(128),
   algorithm character varying(128),
   featureset character varying(128),
   audio_length_seconds integer,
   audio_sampling_rate integer,
   "timestamp" double precision,
   mean_squared_error double precision,
   r2 double precision,
   explained_variance_score double precision,
   num_features integer,
   num_features_below_005_pval integer
)
WITH (
  OIDS = FALSE
)
;